import subprocess


for index in range(1000):
    cmd = "python detector.py --images ../../data/kaggle/challenge2018_test/{} --out out/images".format(index)
    print("Batch {} started".format(index))
    subprocess.check_output(cmd, shell=True)
