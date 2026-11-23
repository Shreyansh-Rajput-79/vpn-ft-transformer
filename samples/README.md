# Sample captures

Two short packet captures from the VNAT dataset (MIT Lincoln Laboratory, see the main README
for the citation), included so the project can be tried without downloading VNAT:

| File | Traffic | Flows |
|---|---|---|
| `vpn_skype-chat_capture11.pcap` | Skype chat through OpenVPN (UDP) | 230 |
| `nonvpn_skype-chat_capture13.pcap` | Skype chat without a VPN | 231 |

Both are in the **test split** (seed 42) of the shipped models: the models never saw them
during training, so the results on them are a fair example.

```bash
python predict.py samples/vpn_skype-chat_capture11.pcap -m models/vnat_noflags_model.pth --explain --trace 1
```
