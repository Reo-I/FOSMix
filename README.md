# FOSMix
Frequency-based Optimal Style Mix

## Step 0
Install packages
```
$ pip install pipenv
$ virtualenv -p ~/.pyenv/versions/3.10.4/bin/python ~/venvs/fosmix
$ source ~/venvs/fosmix/bin/activate
$ (fosmix) pip install -r requirements.txt
```

## Step 1
Modify the installed packages

```bash
$ (fosmix) chmod 777 modify_package_contents.sh
$ (fosmix) ./modify_package_contents.sh
```


# Argument Description

Hyper parameters with * is required. 

1. **--dataset** (str*) : Select the dataset.
    - OEM (OpenEarthMap)
    - FLAIR

2. **--n_epochs** (int*): Number of training epochs.
    - 150 (for OEM dataset)
    - 50 (for FLAIR dataset)

3. **--ver** (int*): Version.

4. **--final** (bool): Use the final model parameters for testing.
    - 0 (Use the parameters that gave the best results on the validation data for testing)
    - 1 (Use the last parameters)

5. **--randomize** (bool*): Randomize images ot not, i.e., whether to use the baseline or the proposal.
    - 0 (baseline)
    - 1 (use the proposal)

6. **--optimize** (bool*): Optimize the mask or not. When `optimize` is 1, `randomize` is always 1.
    - 0 (baseline or FULL MIX)
    - 1 (OPTIMAL MIX)

7. **--aug_color** (float): Probability of color change in augmentation.

8. **--MFI** (int*): Mask From Image or not, whether to generate the OPTIMAL MASK from the image.
    - 0 (Learn and use one mask for all images)
    - 1 (Generate from the image)
  
9. **--fullmask** (int*): Use FULL MIX or not. You can Also use the `optimize` together with this option. If `fullmask` is 1, `randomize` must always be set to 1 as well.
    - 0 (Do not use FULL MIX)
    - 1 (Use FULL MIX)

