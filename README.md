# colorization819

Code for Amanda Li and Handong Wang's 6.819 final project

## To run colorization models

1. Download repo and download ADE20k scene parsing benchmark dataset here: http://sceneparsing.csail.mit.edu/

2. Move downloaded `data/ADEChallengeData2016` to `colorization` directory.

3. If not on AWS EC2 instance, change any file paths prefixed with `/home/ec2-user/` to path to this repo.

4. Run `python ade2k_infer_prior_ab.py` to generate inferred ab channels per image.

5. Modify model params (`input_B, mask_B`) in `ade2k_dataset.py` if needed, then run `python ade2k_dataset.py` to train and evaluate models.
