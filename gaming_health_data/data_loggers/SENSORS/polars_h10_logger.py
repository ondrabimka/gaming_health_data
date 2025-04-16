# Copied from: https://github.com/fsmeraldi/bleakheart/blob/main/examples/ecg_queue.py
""" 
This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

Copyright (C) 2023 Fabrizio Smeraldi <fabrizio@smeraldi.net>
"""

""" ECG acquisition using an asynchronous queue - producer/consumer model """

import sys
import asyncio
from bleak import BleakScanner, BleakClient
from bleakheart import PolarMeasurementData 
from threading import Thread
from datetime import datetime
import matplotlib.pyplot as plt
import argparse
import os

# Due to asyncio limitations on Windows, one cannot use loop.add_reader
# to handle keyboard input; we use threading instead. See
# https://docs.python.org/3.11/library/asyncio-platforms.html
if sys.platform == "win32":
    from threading import Thread
    add_reader_support=False
else:
    add_reader_support=True
    
async def scan():
    """ Scan for a Polar device. """
    device = await BleakScanner.find_device_by_filter(
        lambda dev, adv: dev.name and "polar" in dev.name.lower())
    return device


async def run_ble_client(device, queue):
    """ This task connects to the BLE server (the heart rate sensor)
    identified by device, starts ECG notification and pushes the ECG 
    data to the queue. The tasks terminates when the sensor disconnects 
    or the user hits enter. """

    def keyboard_handler(loop=None):
        """ Called by the asyncio loop when the user hits Enter, 
        or run in a separate thread (if no add_reader support). In 
        this case, the event loop is passed as an argument """
        input() # clear input buffer
        print (f"Quitting on user command")
        # causes the client task to exit
        if loop == None:
            quitclient.set() # we are in the event loop thread
        else:
            # we are in a separate thread - call set in the event loop thread
            loop.call_soon_threadsafe(quitclient.set)
    
    def disconnected_callback(client):
        """ Called by BleakClient if the sensor disconnects """
        print("Sensor disconnected")
        quitclient.set() # causes the ble client task to exit

    # we use this event to signal the end of the client task
    quitclient = asyncio.Event()
    print(f"Connecting to {device}...")
    # the context manager will handle connection/disconnection for us
    async with BleakClient(device, disconnected_callback=
                           disconnected_callback) as client:
        print(f"Connected: {client.is_connected}")
        # create the Polar Measurement Data object; set queue for
        # ecg data
        pmd = PolarMeasurementData(client, ecg_queue=queue)
        # ask about ACC settings
        settings = await pmd.available_settings('ECG')
        print("Request for available ECG settings returned the following:")
        for k,v in settings.items():
            print(f"{k}:\t{v}")
        loop=asyncio.get_running_loop()
        if add_reader_support:
            # Set the loop to call keyboard_handler when one line of input is
            # ready on stdin
            loop.add_reader(sys.stdin, keyboard_handler)
        else:
            # run keyboard_handler in a daemon thread
            Thread(target=keyboard_handler, kwargs={'loop': loop},
                   daemon=True).start()
        print(">>> Hit Enter to exit <<<")
        # start notifications; bleakheart will start pushing
        # data to the queue we passed to PolarMeasurementData
        (err_code, err_msg, _)= await pmd.start_streaming('ECG')
        if err_code != 0:
            print(f"PMD returned an error: {err_msg}")
            sys.exit(err_code)
        # this task does not need to do anything else; wait until
        # user hits enter or the sensor disconnects
        await quitclient.wait()
        # no need to stop notifications if we are exiting the context
        # manager anyway, as they will disconnect the client; however,
        # it's easy to stop them if we want to
        if client.is_connected:
            await pmd.stop_streaming('ECG')
        if add_reader_support:
            loop.remove_reader(sys.stdin)
        # signal the consumer task to quit
        queue.put_nowait(('QUIT', None, None, None))


async def run_consumer_task(queue, save_data=True, plot_data=False):
    print("After connecting, will process ECG data")
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    filename = f'gaming_health_data/recorded_data/SENSORS/ecg_data_polars_h10_{timestamp}.txt'
    
    # Setup plotting if enabled
    if plot_data:
        plt.ion()  # Enable interactive mode
        fig, ax = plt.subplots()
        line, = ax.plot([], [])
        ax.set_title('ECG Data')
        ax.set_xlabel('Samples')
        ax.set_ylabel('Voltage (μV)')
        data_buffer = []
        MAX_POINTS = 500  # Number of points to display

    if save_data:
        if not os.path.exists('gaming_health_data/recorded_data/SENSORS'):
            os.makedirs('gaming_health_data/recorded_data/SENSORS')
        f = open(filename, 'w')
    
    try:
        while True:
            frame = await queue.get()
            if frame[0] == 'QUIT':
                break
                
            if save_data:
                f.write(f"{frame}\n")
                f.flush()
                
            if plot_data and frame[0] == 'ECG':
                # Add new data points
                data_buffer.extend(frame[2])
                if len(data_buffer) > MAX_POINTS:
                    data_buffer = data_buffer[-MAX_POINTS:]
                
                # Update plot
                line.set_data(range(len(data_buffer)), data_buffer)
                ax.relim()
                ax.autoscale_view()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                
            print(frame)
    finally:
        if save_data:
            f.close()
        if plot_data:
            plt.close()


async def main(save_data = True, plot_data = False):
    print("Scanning for BLE devices")
    device = await scan()
    if device == None:
        print("Polar device not found.")
        sys.exit(-4)
    # the queue needs to be long enough to cache all the frames, since
    # PolarMeasurementData uses put_nowait
    ecgqueue = asyncio.Queue()
    # producer task will return when the user hits enter or the
    # sensor disconnects
    producer = run_ble_client(device, ecgqueue)
    consumer = run_consumer_task(ecgqueue, save_data=save_data, plot_data=plot_data)
    # wait for the two tasks to exit
    await asyncio.gather(producer, consumer)
    print("Bye.")


if __name__ == "__main__":    
    parser = argparse.ArgumentParser(description="Record ECG data from a Polar device")
    parser.add_argument('-s', '--save', action='store_true', help="Save data to file", default=True)
    parser.add_argument('-p', '--plot', action='store_true', help="Plot data in real time", default=False)
    args = parser.parse_args()
    # run the main function with the provided arguments
    asyncio.run(main(save_data=args.save, plot_data=args.plot))

    # example usage:
    # python gaming_health_data/data_loggers/SENSORS/polars_h10_logger.py --save --plot