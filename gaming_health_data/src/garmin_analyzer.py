from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class NumpyEncoder(json.JSONEncoder):
	"""JSON encoder that safely serializes numpy and pandas scalar/date types."""

	def default(self, obj):
		if isinstance(obj, (np.integer, np.floating, np.bool_)):
			return obj.item()
		if isinstance(obj, np.ndarray):
			return obj.tolist()
		if isinstance(obj, (datetime, pd.Timestamp)):
			return obj.isoformat()
		if pd.isna(obj):
			return None
		return super().default(obj)


@pd.api.extensions.register_dataframe_accessor("garmin")
class GarminAnalyzer:
	"""
	Comprehensive Garmin FIT analyzer as pandas extension.

	Provides both compatibility methods used by Apple Watch/Oura pipelines and
	higher-level summary/analysis outputs suitable for app/API usage.
	"""

	def __init__(self, pandas_obj):
		self._obj = pandas_obj
		self._data_cache = {}
		self._analysis_cache = {}

	@classmethod
	def read_fit_data(cls, data_path: Union[str, Path]) -> pd.DataFrame:
		"""
		Read Garmin FIT file(s) and return record-level dataframe.

		Additional parsed tables are attached under `df.attrs["garmin_data"]`.
		"""
		data_path = Path(data_path)

		if data_path.is_dir():
			fit_files = sorted(data_path.glob("*.fit"))
		elif data_path.is_file() and data_path.suffix.lower() == ".fit":
			fit_files = [data_path]
		else:
			raise FileNotFoundError(f"No .fit file found at: {data_path}")

		if not fit_files:
			print(f"No .fit files found in {data_path}")
			return pd.DataFrame()

		all_records: List[pd.DataFrame] = []
		all_sessions: List[pd.DataFrame] = []
		all_laps: List[pd.DataFrame] = []
		all_activities: List[pd.DataFrame] = []
		parse_errors: List[Tuple[str, str]] = []

		for fit_file in fit_files:
			try:
				parsed = cls._parse_fit_file(fit_file)
				if not parsed["records"].empty:
					all_records.append(parsed["records"])
				if not parsed["sessions"].empty:
					all_sessions.append(parsed["sessions"])
				if not parsed["laps"].empty:
					all_laps.append(parsed["laps"])
				if not parsed["activities"].empty:
					all_activities.append(parsed["activities"])
				print(
					f"Loaded {fit_file.name}: "
					f"{len(parsed['records'])} records, "
					f"{len(parsed['sessions'])} sessions"
				)
			except Exception as e:
				parse_errors.append((str(fit_file), str(e)))

		records_df = pd.concat(all_records, ignore_index=True) if all_records else pd.DataFrame()
		sessions_df = pd.concat(all_sessions, ignore_index=True) if all_sessions else pd.DataFrame()
		laps_df = pd.concat(all_laps, ignore_index=True) if all_laps else pd.DataFrame()
		activities_df = pd.concat(all_activities, ignore_index=True) if all_activities else pd.DataFrame()

		records_df = cls._normalize_time_columns(records_df, ["timestamp"])
		sessions_df = cls._normalize_time_columns(sessions_df, ["start_time", "timestamp"])
		laps_df = cls._normalize_time_columns(laps_df, ["start_time", "timestamp"])
		activities_df = cls._normalize_time_columns(activities_df, ["timestamp", "local_timestamp"])

		if not records_df.empty and "timestamp" in records_df.columns:
			records_df = records_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
			records_df["date"] = records_df["timestamp"].dt.date

		records_df.attrs["garmin_data"] = {
			"sessions": sessions_df,
			"laps": laps_df,
			"activities": activities_df,
			"files": [str(f) for f in fit_files],
			"errors": parse_errors,
		}

		if parse_errors:
			print(f"[WARNING] {len(parse_errors)} file(s) had parsing errors")
			for file_path, error in parse_errors:
				print(f"  - {Path(file_path).name}: {error}")

		print(f"Combined Garmin dataset: {len(records_df)} records from {len(fit_files)} file(s)")
		return records_df

	@classmethod
	def read_file(cls, file_path: Union[str, Path]) -> pd.DataFrame:
		"""Alias for `read_fit_data` (API consistency with other analyzers)."""
		return cls.read_fit_data(file_path)

	@staticmethod
	def _normalize_time_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
		"""Normalize datetime-like columns to pandas datetime (coerce invalid)."""
		if df.empty:
			return df
		for col in columns:
			if col in df.columns:
				df[col] = pd.to_datetime(df[col], errors="coerce")
		return df

	@staticmethod
	def _parse_fit_file(file_path: Path) -> Dict[str, pd.DataFrame]:
		"""Parse one FIT file using fitparse and return known message tables."""
		try:
			from fitparse import FitFile
		except Exception as e:
			raise ImportError(
				"Missing dependency `fitparse`. Install it with: pip install fitparse"
			) from e

		fit = FitFile(str(file_path))

		def _collect(message_name: str) -> pd.DataFrame:
			rows = []
			for message in fit.get_messages(message_name):
				row = message.get_values()
				row["source_file"] = file_path.name
				rows.append(row)
			return pd.DataFrame(rows)

		return {
			"records": _collect("record"),
			"sessions": _collect("session"),
			"laps": _collect("lap"),
			"activities": _collect("activity"),
		}

	def _get_data(self, data_type: str) -> pd.DataFrame:
		"""Get parsed table from attrs/cache."""
		if data_type == "records":
			return self._obj

		if data_type in self._data_cache:
			return self._data_cache[data_type]

		if hasattr(self._obj, "attrs") and "garmin_data" in self._obj.attrs:
			garmin_data = self._obj.attrs["garmin_data"]
			if data_type in garmin_data:
				self._data_cache[data_type] = garmin_data[data_type]
				return garmin_data[data_type]

		return pd.DataFrame()

	@staticmethod
	def _resolve_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
		for candidate in candidates:
			if candidate in df.columns:
				return candidate
		return None

	@staticmethod
	def _resolve_hr_column(df: pd.DataFrame) -> Optional[str]:
		"""Find heart-rate column in Garmin records."""
		return GarminAnalyzer._resolve_column(df, ["heart_rate", "enhanced_heart_rate", "bpm", "hr"])

	@staticmethod
	def _apply_date_range(df: pd.DataFrame, time_col: str, date_range: Optional[Tuple[str, str]]) -> pd.DataFrame:
		if df.empty or time_col not in df.columns or not date_range:
			return df
		start_date, end_date = date_range
		start_dt = pd.to_datetime(start_date)
		end_dt = pd.to_datetime(end_date)
		return df[(df[time_col] >= start_dt) & (df[time_col] <= end_dt)]

	def get_available_metrics(self) -> List[str]:
		"""Get available Garmin tables/metrics loaded from FIT data."""
		available = []
		if not self._obj.empty:
			available.append("records")
		if hasattr(self._obj, "attrs") and "garmin_data" in self._obj.attrs:
			for key, value in self._obj.attrs["garmin_data"].items():
				if isinstance(value, pd.DataFrame) and not value.empty:
					available.append(key)
		return available

	# === Compatibility methods (used by existing multimodal pipeline) ===

	def get_heart_rate_for_date(self, target_date: str) -> pd.DataFrame:
		"""Get Garmin HR for one date. Returns timestamp/bpm/value/time_seconds."""
		records = self._get_data("records")
		if records.empty or "timestamp" not in records.columns:
			return pd.DataFrame()

		hr_col = self._resolve_hr_column(records)
		if hr_col is None:
			return pd.DataFrame()

		df = records.copy()
		df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
		df[hr_col] = pd.to_numeric(df[hr_col], errors="coerce")
		df = df.dropna(subset=["timestamp", hr_col])
		target_date_obj = pd.to_datetime(target_date).date()
		df = df[df["timestamp"].dt.date == target_date_obj].copy()

		if df.empty:
			return pd.DataFrame()

		start_time = df["timestamp"].min()
		df["time_seconds"] = (df["timestamp"] - start_time).dt.total_seconds()
		df["bpm"] = df[hr_col]
		df["value"] = df[hr_col]

		return df[["timestamp", "bpm", "value", "time_seconds"]].sort_values("timestamp").reset_index(drop=True)

	def get_session_from_date(self, target_date: str) -> Optional[dict]:
		"""Get inferred Garmin session for a date (session table first, HR fallback)."""
		target_date_obj = pd.to_datetime(target_date).date()
		sessions = self._get_data("sessions")

		if not sessions.empty:
			candidate_cols = [c for c in ["start_time", "timestamp"] if c in sessions.columns]
			if candidate_cols:
				s = sessions.copy()
				for col in candidate_cols:
					s[col] = pd.to_datetime(s[col], errors="coerce")
				s = s.dropna(subset=[candidate_cols[0]])
				day_sessions = s[s[candidate_cols[0]].dt.date == target_date_obj]

				if not day_sessions.empty:
					if "total_elapsed_time" in day_sessions.columns:
						day_sessions = day_sessions.copy()
						day_sessions["total_elapsed_time"] = pd.to_numeric(day_sessions["total_elapsed_time"], errors="coerce")
						session = day_sessions.sort_values("total_elapsed_time", ascending=False).iloc[0]
					else:
						session = day_sessions.iloc[0]

					start_date = pd.to_datetime(session.get("start_time", session.get("timestamp")), errors="coerce")
					elapsed_s = pd.to_numeric(session.get("total_elapsed_time", None), errors="coerce")
					if pd.notna(elapsed_s):
						end_date = start_date + pd.to_timedelta(float(elapsed_s), unit="s")
					else:
						end_date = pd.to_datetime(session.get("timestamp", start_date), errors="coerce")

					return {
						"startDate": start_date,
						"endDate": end_date,
						"date": str(target_date),
						"source": "Garmin",
						"sport": session.get("sport"),
						"sub_sport": session.get("sub_sport"),
					}

		hr_data = self.get_heart_rate_for_date(target_date)
		if hr_data.empty:
			return None

		return {
			"startDate": hr_data["timestamp"].min(),
			"endDate": hr_data["timestamp"].max(),
			"date": str(target_date),
			"source": "Garmin",
			"sport": None,
			"sub_sport": None,
		}

	def get_heart_rate_stats_from_session(self, session: dict) -> pd.DataFrame:
		"""Get Garmin HR constrained to session. Returns timestamp/bpm/value/time_seconds."""
		if not session or "startDate" not in session or "endDate" not in session:
			return pd.DataFrame()

		records = self._get_data("records")
		if records.empty or "timestamp" not in records.columns:
			return pd.DataFrame()

		hr_col = self._resolve_hr_column(records)
		if hr_col is None:
			return pd.DataFrame()

		start_date = pd.to_datetime(session["startDate"])
		end_date = pd.to_datetime(session["endDate"])

		df = records.copy()
		df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
		df[hr_col] = pd.to_numeric(df[hr_col], errors="coerce")
		df = df.dropna(subset=["timestamp", hr_col])
		df = df[(df["timestamp"] >= start_date) & (df["timestamp"] <= end_date)].copy()

		if df.empty:
			return pd.DataFrame()

		df["time_seconds"] = (df["timestamp"] - start_date).dt.total_seconds()
		df["bpm"] = df[hr_col]
		df["value"] = df[hr_col]

		return df[["timestamp", "bpm", "value", "time_seconds"]].sort_values("timestamp").reset_index(drop=True)

	def get_measurement_period_for_date(self, target_date: str) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
		"""Get first/last Garmin HR measurement timestamps for a date."""
		hr_data = self.get_heart_rate_for_date(target_date)
		if hr_data.empty:
			return None, None
		return hr_data["timestamp"].min(), hr_data["timestamp"].max()

	@property
	def measure_start_date(self):
		"""Earliest Garmin measurement timestamp."""
		if self._obj.empty or "timestamp" not in self._obj.columns:
			return None
		ts = pd.to_datetime(self._obj["timestamp"], errors="coerce").dropna()
		return ts.min() if not ts.empty else None

	@property
	def measure_end_date(self):
		"""Latest Garmin measurement timestamp."""
		if self._obj.empty or "timestamp" not in self._obj.columns:
			return None
		ts = pd.to_datetime(self._obj["timestamp"], errors="coerce").dropna()
		return ts.max() if not ts.empty else None

	# === Comprehensive analysis methods (app/API friendly) ===

	def analyze_heart_rate(self, date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
		"""Comprehensive Garmin heart-rate analysis."""
		records = self._get_data("records")
		hr_col = self._resolve_hr_column(records)
		if records.empty or hr_col is None or "timestamp" not in records.columns:
			return {"error": "No heart rate data available"}

		h_df = records[["timestamp", hr_col]].copy()
		h_df["timestamp"] = pd.to_datetime(h_df["timestamp"], errors="coerce")
		h_df[hr_col] = pd.to_numeric(h_df[hr_col], errors="coerce")
		h_df = h_df.dropna(subset=["timestamp", hr_col]).rename(columns={hr_col: "bpm"})
		h_df = self._apply_date_range(h_df, "timestamp", date_range)

		if h_df.empty:
			return {"error": "No heart rate data available for selected range"}

		analysis: Dict[str, Any] = {
			"basic_stats": {
				"mean_bpm": h_df["bpm"].mean(),
				"median_bpm": h_df["bpm"].median(),
				"std_bpm": h_df["bpm"].std(),
				"min_bpm": h_df["bpm"].min(),
				"max_bpm": h_df["bpm"].max(),
				"total_recordings": len(h_df),
			},
			"percentiles": {f"p{p}": h_df["bpm"].quantile(p / 100) for p in [5, 10, 25, 50, 75, 90, 95]},
		}

		max_hr = 190
		analysis["zones"] = {
			"zone1_50_60": len(h_df[(h_df["bpm"] >= max_hr * 0.5) & (h_df["bpm"] < max_hr * 0.6)]) / len(h_df) * 100,
			"zone2_60_70": len(h_df[(h_df["bpm"] >= max_hr * 0.6) & (h_df["bpm"] < max_hr * 0.7)]) / len(h_df) * 100,
			"zone3_70_80": len(h_df[(h_df["bpm"] >= max_hr * 0.7) & (h_df["bpm"] < max_hr * 0.8)]) / len(h_df) * 100,
			"zone4_80_90": len(h_df[(h_df["bpm"] >= max_hr * 0.8) & (h_df["bpm"] < max_hr * 0.9)]) / len(h_df) * 100,
			"zone5_90_max": len(h_df[h_df["bpm"] >= max_hr * 0.9]) / len(h_df) * 100,
		}

		h_df["hour"] = h_df["timestamp"].dt.hour
		h_df["day_of_week"] = h_df["timestamp"].dt.day_name()
		analysis["temporal_patterns"] = {
			"hourly_avg": h_df.groupby("hour")["bpm"].mean().to_dict(),
			"daily_avg": h_df.groupby("day_of_week")["bpm"].mean().to_dict(),
		}

		rr_intervals = 60000.0 / h_df.sort_values("timestamp")["bpm"]
		successive_diffs = np.diff(rr_intervals)
		rmssd = float(np.sqrt(np.mean(successive_diffs**2))) if len(successive_diffs) > 0 else None
		analysis["hrv_estimation"] = {
			"rmssd_ms": rmssd,
			"sdnn_ms": float(np.std(rr_intervals)) if len(rr_intervals) > 0 else None,
			"mean_rr_ms": float(np.mean(rr_intervals)) if len(rr_intervals) > 0 else None,
		}

		return analysis

	def analyze_sessions(self, date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
		"""Analyze Garmin session summaries (duration, distance, calories, sport mix)."""
		sessions = self._get_data("sessions")
		if sessions.empty:
			return {"error": "No session data available"}

		s_df = sessions.copy()
		time_col = self._resolve_column(s_df, ["start_time", "timestamp"])
		if time_col:
			s_df[time_col] = pd.to_datetime(s_df[time_col], errors="coerce")
			s_df = self._apply_date_range(s_df, time_col, date_range)

		if s_df.empty:
			return {"error": "No session data available for selected range"}

		duration_col = self._resolve_column(s_df, ["total_elapsed_time", "total_timer_time"])
		distance_col = self._resolve_column(s_df, ["total_distance", "enhanced_avg_speed"])
		calories_col = self._resolve_column(s_df, ["total_calories", "calories"])

		if duration_col:
			s_df[duration_col] = pd.to_numeric(s_df[duration_col], errors="coerce")
		if distance_col:
			s_df[distance_col] = pd.to_numeric(s_df[distance_col], errors="coerce")
		if calories_col:
			s_df[calories_col] = pd.to_numeric(s_df[calories_col], errors="coerce")

		analysis: Dict[str, Any] = {
			"total_sessions": int(len(s_df)),
			"sport_breakdown": s_df["sport"].value_counts().to_dict() if "sport" in s_df.columns else {},
		}

		if duration_col:
			analysis["duration_minutes"] = {
				"mean": float(s_df[duration_col].mean() / 60),
				"median": float(s_df[duration_col].median() / 60),
				"total": float(s_df[duration_col].sum() / 60),
			}
		if distance_col:
			analysis["distance_meters"] = {
				"mean": float(s_df[distance_col].mean()),
				"total": float(s_df[distance_col].sum()),
			}
		if calories_col:
			analysis["calories"] = {
				"mean": float(s_df[calories_col].mean()),
				"total": float(s_df[calories_col].sum()),
			}

		return analysis

	def analyze_activity(self, date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
		"""Analyze record-level activity metrics such as speed, cadence, distance, power."""
		records = self._get_data("records")
		if records.empty or "timestamp" not in records.columns:
			return {"error": "No activity record data available"}

		r_df = records.copy()
		r_df["timestamp"] = pd.to_datetime(r_df["timestamp"], errors="coerce")
		r_df = r_df.dropna(subset=["timestamp"])
		r_df = self._apply_date_range(r_df, "timestamp", date_range)

		if r_df.empty:
			return {"error": "No activity data available for selected range"}

		metrics = {
			"distance": ["distance"],
			"speed": ["enhanced_speed", "speed"],
			"cadence": ["cadence", "enhanced_avg_cadence"],
			"power": ["power"],
			"altitude": ["enhanced_altitude", "altitude"],
			"temperature": ["temperature"],
		}

		analysis: Dict[str, Any] = {"basic_stats": {}}
		for metric_name, candidates in metrics.items():
			col = self._resolve_column(r_df, candidates)
			if col:
				r_df[col] = pd.to_numeric(r_df[col], errors="coerce")
				valid = r_df[col].dropna()
				if not valid.empty:
					analysis["basic_stats"][metric_name] = {
						"mean": float(valid.mean()),
						"median": float(valid.median()),
						"min": float(valid.min()),
						"max": float(valid.max()),
					}

		return analysis if analysis["basic_stats"] else {"error": "No compatible activity metric columns found"}

	def _calculate_cross_correlations(self, date_range: Optional[Tuple[str, str]] = None) -> Dict[str, float]:
		"""Calculate basic cross-metric correlations from record-level data."""
		records = self._get_data("records")
		if records.empty:
			return {}

		df = records.copy()
		if "timestamp" in df.columns:
			df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
			df = df.dropna(subset=["timestamp"])
			df = self._apply_date_range(df, "timestamp", date_range)

		hr_col = self._resolve_hr_column(df)
		speed_col = self._resolve_column(df, ["enhanced_speed", "speed"])
		cadence_col = self._resolve_column(df, ["cadence", "enhanced_avg_cadence"])
		power_col = self._resolve_column(df, ["power"])

		corrs = {}
		for col, name in [(speed_col, "hr_vs_speed"), (cadence_col, "hr_vs_cadence"), (power_col, "hr_vs_power")]:
			if hr_col and col:
				tmp = df[[hr_col, col]].apply(pd.to_numeric, errors="coerce").dropna()
				if len(tmp) >= 3:
					corrs[name] = float(tmp[hr_col].corr(tmp[col]))

		return corrs

	def _generate_health_insights(self, analysis: Dict[str, Any]) -> List[str]:
		"""Generate concise Garmin-focused insights."""
		insights = []

		hr = analysis.get("heart_rate", {})
		if "basic_stats" in hr:
			mean_bpm = hr["basic_stats"].get("mean_bpm")
			if mean_bpm is not None:
				if mean_bpm > 100:
					insights.append("Average heart rate is elevated; consider balancing intense sessions with recovery.")
				elif mean_bpm < 50:
					insights.append("Average heart rate is low; this can be normal for trained athletes.")

		sessions = analysis.get("sessions", {})
		if "total_sessions" in sessions and sessions["total_sessions"] > 0:
			insights.append(f"Tracked {sessions['total_sessions']} Garmin session(s) in the selected period.")

		if not insights:
			insights.append("Continue tracking more sessions to build stronger trends and recommendations.")

		return insights

	def total_analysis(self, date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
		"""Comprehensive Garmin analysis across heart rate, sessions, and activity."""
		analysis = {
			"analysis_date": datetime.now().isoformat(),
			"date_range": date_range,
			"available_metrics": self.get_available_metrics(),
		}

		analysis["heart_rate"] = self.analyze_heart_rate(date_range)
		analysis["sessions"] = self.analyze_sessions(date_range)
		analysis["activity"] = self.analyze_activity(date_range)
		analysis["correlations"] = self._calculate_cross_correlations(date_range)
		analysis["insights"] = self._generate_health_insights(analysis)

		return analysis

	def day_specific_analysis(self, target_date: str) -> Dict[str, Any]:
		"""Detailed Garmin analysis for one date."""
		target_date_obj = pd.to_datetime(target_date).date()
		start = pd.Timestamp(target_date_obj)
		end = start + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)

		analysis = {
			"date": str(target_date_obj),
			"analysis_timestamp": datetime.now().isoformat(),
			"session": self.get_session_from_date(str(target_date_obj)),
			"heart_rate": self.analyze_heart_rate((str(start), str(end))),
			"activity": self.analyze_activity((str(start), str(end))),
		}

		return analysis

	def get_comprehensive_summary(self, date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
		"""Generate app/chatbot friendly Garmin summary."""
		analysis = self.total_analysis(date_range)

		hr_stats = analysis.get("heart_rate", {}).get("basic_stats", {})
		sessions = analysis.get("sessions", {})

		score_components = []
		if hr_stats.get("mean_bpm") is not None:
			mean_bpm = hr_stats["mean_bpm"]
			# simple normalization around moderate zone
			hr_score = max(0, 100 - abs(75 - mean_bpm) * 1.5)
			score_components.append(hr_score)
		if sessions.get("total_sessions"):
			session_score = min(100, sessions["total_sessions"] * 20)
			score_components.append(session_score)

		wellness_score = float(np.mean(score_components)) if score_components else None
		overall_status = (
			"Excellent" if wellness_score is not None and wellness_score >= 85 else
			"Good" if wellness_score is not None and wellness_score >= 75 else
			"Fair" if wellness_score is not None and wellness_score >= 65 else
			"Needs attention" if wellness_score is not None else "Insufficient data"
		)

		return {
			"overview": {
				"report_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				"analysis_period": {
					"start_date": date_range[0] if date_range else "All available data",
					"end_date": date_range[1] if date_range else "All available data",
				},
				"data_sources": analysis.get("available_metrics", []),
				"overall_health_status": overall_status,
			},
			"key_metrics": {
				"mean_hr_bpm": hr_stats.get("mean_bpm"),
				"max_hr_bpm": hr_stats.get("max_bpm"),
				"min_hr_bpm": hr_stats.get("min_bpm"),
				"total_recordings": hr_stats.get("total_recordings"),
				"total_sessions": sessions.get("total_sessions"),
			},
			"health_insights": {
				"personalized_recommendations": analysis.get("insights", []),
			},
			"wellness_score": {
				"score": round(wellness_score, 1) if wellness_score is not None else None,
				"status": overall_status,
			},
			"raw_analysis": analysis,
		}

	def to_json(
		self,
		date_range: Optional[Tuple[str, str]] = None,
		include_raw_data: bool = False,
		pretty_print: bool = True,
	) -> str:
		"""Export Garmin analysis as JSON string."""
		payload = self.total_analysis(date_range)
		if include_raw_data and not self._obj.empty:
			payload["raw_records_preview"] = self._obj.head(200).to_dict("records")
		indent = 2 if pretty_print else None
		return json.dumps(payload, cls=NumpyEncoder, indent=indent)

	def to_api_response(self, date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
		"""Generate API-ready Garmin response format."""
		try:
			summary = self.get_comprehensive_summary(date_range)
			return {
				"status": "success",
				"message": "Garmin analysis generated successfully",
				"data": summary,
				"metadata": {
					"generated_at": datetime.now().isoformat(),
					"source": "Garmin FIT",
				},
			}
		except Exception as e:
			return {
				"status": "error",
				"message": str(e),
				"data": None,
				"metadata": {"generated_at": datetime.now().isoformat()},
			}

	def to_chatbot_context(self, date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
		"""Generate concise chatbot context from Garmin data."""
		summary = self.get_comprehensive_summary(date_range)
		return {
			"user_health_profile": {
				"overall_status": summary["overview"]["overall_health_status"],
				"wellness_score": summary["wellness_score"]["score"],
			},
			"recent_metrics": summary["key_metrics"],
			"actionable_insights": summary["health_insights"]["personalized_recommendations"],
			"data_freshness": {
				"last_analysis": summary["overview"]["report_date"],
				"data_period": summary["overview"]["analysis_period"],
			},
		}

	def plot_comprehensive_dashboard(self, date_range: Optional[Tuple[str, str]] = None):
		"""Render a simple interactive Garmin dashboard with key visual panels."""
		fig = make_subplots(
			rows=2,
			cols=2,
			subplot_titles=(
				"Heart Rate Over Time",
				"Heart Rate Distribution",
				"Speed Over Time",
				"Session Durations (min)",
			),
		)

		records = self._get_data("records")
		sessions = self._get_data("sessions")

		if not records.empty and "timestamp" in records.columns:
			r = records.copy()
			r["timestamp"] = pd.to_datetime(r["timestamp"], errors="coerce")
			r = r.dropna(subset=["timestamp"])
			r = self._apply_date_range(r, "timestamp", date_range)

			hr_col = self._resolve_hr_column(r)
			if hr_col:
				r[hr_col] = pd.to_numeric(r[hr_col], errors="coerce")
				h = r.dropna(subset=[hr_col])
				if not h.empty:
					fig.add_trace(go.Scatter(x=h["timestamp"], y=h[hr_col], mode="lines", name="HR"), row=1, col=1)
					fig.add_trace(go.Histogram(x=h[hr_col], name="HR dist", nbinsx=30), row=1, col=2)

			speed_col = self._resolve_column(r, ["enhanced_speed", "speed"])
			if speed_col:
				r[speed_col] = pd.to_numeric(r[speed_col], errors="coerce")
				s = r.dropna(subset=[speed_col])
				if not s.empty:
					fig.add_trace(go.Scatter(x=s["timestamp"], y=s[speed_col], mode="lines", name="Speed"), row=2, col=1)

		if not sessions.empty:
			s = sessions.copy()
			duration_col = self._resolve_column(s, ["total_elapsed_time", "total_timer_time"])
			label_col = self._resolve_column(s, ["sport", "sub_sport", "source_file"])
			if duration_col:
				s[duration_col] = pd.to_numeric(s[duration_col], errors="coerce") / 60.0
				s = s.dropna(subset=[duration_col])
				if not s.empty:
					labels = s[label_col] if label_col else [f"session_{i}" for i in range(len(s))]
					fig.add_trace(go.Bar(x=labels, y=s[duration_col], name="Duration (min)"), row=2, col=2)

		fig.update_layout(title_text="Garmin Comprehensive Dashboard", height=800, showlegend=True)
		fig.show()

	def export_analysis_report(
		self,
		output_path: str,
		date_range: Optional[Tuple[str, str]] = None,
		report_type: str = "full",
	):
		"""Export Garmin analysis as .json or .csv."""
		output = Path(output_path)

		if output.suffix.lower() == ".json":
			if report_type == "summary":
				payload = self.get_comprehensive_summary(date_range)
			else:
				payload = self.total_analysis(date_range)
			output.write_text(json.dumps(payload, cls=NumpyEncoder, indent=2), encoding="utf-8")
			print(f"Garmin JSON report exported to {output}")

		elif output.suffix.lower() == ".csv":
			records = self._get_data("records")
			if records.empty:
				raise ValueError("No Garmin records available for CSV export")
			to_export = records.copy()
			if "timestamp" in to_export.columns and date_range:
				to_export["timestamp"] = pd.to_datetime(to_export["timestamp"], errors="coerce")
				to_export = self._apply_date_range(to_export, "timestamp", date_range)
			to_export.to_csv(output, index=False)
			print(f"Garmin CSV exported to {output}")

		else:
			raise ValueError("Unsupported export format. Use .json or .csv")

