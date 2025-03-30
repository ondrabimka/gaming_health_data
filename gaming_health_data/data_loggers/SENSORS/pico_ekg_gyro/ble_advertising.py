import struct

def advertising_payload(limited_disc=False, br_edr=False, name=None, services=None, appearance=0):
    """
    Create a BLE advertising payload.
    
    Args:
        limited_disc (bool): Limited discoverable mode
        br_edr (bool): BR/EDR (Basic Rate/Enhanced Data Rate) support
        name (str): Device name to advertise
        services (list): List of service UUIDs to advertise
        appearance (int): Device appearance value
    
    Returns:
        bytearray: Formatted BLE advertising payload
    """
    payload = bytearray()
    
    def _append(adv_type, value):
        """
        Helper function to append data to the payload
        
        Args:
            adv_type (int): Advertising data type
            value (bytes): Value to append
        """
        nonlocal payload
        payload += struct.pack("!BB", len(value) + 1, adv_type) + value
    
    # Flags
    _append(
        0x01,  # Flags AD type
        struct.pack("B", 
            (0x01 if limited_disc else 0x02) +  # Discoverable mode
            (0x18 if br_edr else 0x04)  # BR/EDR support
        ),
    )
    
    # Name
    if name:
        _append(0x09, name.encode())
    
    # Services
    if services:
        for uuid in services:
            b = bytes(uuid)
            if len(b) == 2:
                _append(0x03, b)  # 16-bit UUID
            elif len(b) == 4:
                _append(0x05, b)  # 32-bit UUID
            elif len(b) == 16:
                _append(0x07, b)  # 128-bit UUID
    
    # Appearance
    if appearance:
        _append(0x19, struct.pack("<h", appearance))
    
    return payload