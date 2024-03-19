Please execute the following to store preprocessed data in the data folder. Make sure to set the path (hardcoded) - 

```
python3 scripts/main.py
```


To get the trajectory and landmarks via the slam algorithm execute. Make sure to specify the dataset, and set the output paths (hardcoded) in the file utils/visual_slam. comment or uncomment lines 283-297 as needed. -

```
python3 main.py
```

In order to run dead reckoning, uncomment line 21 in main.py and rerun the above command. To plot the trajectory and landmarks, use - 

```
python3 visualizations.py
```

In order to get the files used in the above file, you need to change the paths to the poses and landmarks obtained by the slam algorithm and dead reckoning. utils/visual_slam_no_correlation.py stores the code for the ablation studies mentioned in the paper.
