# OSiRV Detekcija objekta iz predloška putem normalizirane unakrsne korelacije

[Example Image](output.jpg)

### Installation
Requires python>= 3.6
```bash
pip install -r requirements.txt
```
### For more shapes from smeschke dataset:
```bash
python shapes_download.py
```
### Detect digits
```bash
python normalized_cross_correlation.py
```
### For a different number of displayed images use:
```bash
python normalized_cross_correlation.py   --num_images
```
### For a different number of shapes displayed on image use:
```bash
python normalized_cross_correlation.py   --min_shapes --max_shapes
```