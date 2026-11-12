"""Decision layer of the node firmware (MicroPython on ESP32). Pure Python, no numpy: the same file runs
in the simulator tests and on the microcontroller. Hardware drivers (pulse counter, ultrasonic ranging,
1-Wire thermometer, MOSFET pin) stay outside this module and are written in the prototype phase.

policy.bin layout (little-endian):
    header  'CSTN', version u8, T u16, nx u8, nv u8, k_block u8, n_actions u8,
            v_max_dm3 u16 (cistern capacity in dm3), area_dm2 u16, eff_pct u8,
            theta_fc_permille u16, theta_wp_permille u16,
            actions: n_actions x u8 (target relative depletion x 100)
            taw: T x u16 (root-zone TAW of each crop day, 0.1 mm)
    body    ceil(T / k_block) x nx x nv action indices, one byte each
"""
import struct

MAGIC = b"CSTN"


def pack_policy(table, actions, taw_mm, v_max_m3, area_m2, eff, theta_fc, theta_wp, k_block=1):
    """Build policy.bin from a (T, nx, nv) table of action indices (simulator side)."""
    T, nx, nv = len(table), len(table[0]), len(table[0][0])
    head = MAGIC + struct.pack("<BHBBBBHHBHH", 1, T, nx, nv, k_block, len(actions), min(int(round(v_max_m3 * 1000)), 65535),
                               int(round(area_m2 * 100)), int(round(eff * 100)),
                               int(round(theta_fc * 1000)), int(round(theta_wp * 1000)))
    head += bytes(int(round(a * 100)) for a in actions)
    head += b"".join(struct.pack("<H", int(round(t * 10))) for t in taw_mm)
    body = bytearray()
    for b in range(0, T, k_block):
        for i in range(nx):
            body.extend(bytes(int(v) for v in table[b][i]))
    return head + bytes(body)


class Policy:
    def __init__(self, blob):
        assert blob[:4] == MAGIC
        (self.ver, self.T, self.nx, self.nv, self.k, na, vmax_dm3, area_dm2, eff_pct, fc, wp) = \
            struct.unpack("<BHBBBBHHBHH", blob[4:20])
        self.v_max = vmax_dm3 / 1000.0
        self.area = area_dm2 / 100.0
        self.eff = eff_pct / 100.0
        self.fc, self.wp = fc / 1000.0, wp / 1000.0
        p = 20
        self.actions = [blob[p + i] / 100.0 for i in range(na)]
        p += na
        self.taw = [struct.unpack("<H", blob[p + 2 * i:p + 2 * i + 2])[0] / 10.0 for i in range(self.T)]
        p += 2 * self.T
        self.body = blob[p:]

    def action(self, day, theta, volume_m3):
        """Index of the action for the measured soil water content and cistern volume."""
        x = (self.fc - theta) / (self.fc - self.wp)
        x = 0.0 if x < 0 else (1.0 if x > 1 else x)
        v = volume_m3 / self.v_max
        v = 0.0 if v < 0 else (1.0 if v > 1 else v)
        ix = int(x * (self.nx - 1) + 0.5)
        iv = int(v * (self.nv - 1) + 0.5)
        blk = min(day, self.T - 1) // self.k
        return self.body[(blk * self.nx + ix) * self.nv + iv]

    def litres(self, day, theta, volume_m3):
        """Gross volume (L) to pump today; the pump driver converts it to run time."""
        a = self.action(day, theta, volume_m3)
        x = (self.fc - theta) / (self.fc - self.wp)
        x = 0.0 if x < 0 else (1.0 if x > 1 else x)
        net_mm = max(x - self.actions[a], 0.0) * self.taw[min(day, self.T - 1)]
        return net_mm * self.area / self.eff


def pump_seconds(litres, flow_lpm):
    return int(litres / flow_lpm * 60 + 0.5)
