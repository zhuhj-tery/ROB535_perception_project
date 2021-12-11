# ROB535_perception_project
In this README, we main discuss how to use our classifier. 

Acutually, it's very simple to evaluate the input images. 
1. Clone the repo from github
```bash
git clone https://github.com/zhuhj-tery/ROB535_perception_project.git
```
2. Suppose all the input images are stored in a folder, then the next step is to put the image folder to current directory and rename the folder name to be "test"
3. The final step is the ru the evaluation code
```bash
python3 evaluate.py
```
The the output labels will be recorded in the csv file named "result_test.csv".