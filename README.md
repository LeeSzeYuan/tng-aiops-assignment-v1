# AIOps Engineering Role Assignment Answer

This repository contains the solution for the AIOps Engineering Role Assignment. Below is a breakdown of the folder structure and their contents:
- Please refer to `setupGuide.md` to set up local environment to test run the scripts

## **Question 1 Folder**
- **``README.md``**: Contains answers for the Standard Operating Procedures (SOP).
- **``illustrationCode``**: Includes dummy code that illustrates how to perform some SOPs at a basic level.

## **Question 2 Folder**
- **``data``**: Contains the dataset folder.
- **``src``**: MLFlow python scripts
  - `train-local-filesystem.py`: MLFlow python scripst to log metrics, models, parameters & register models using **local filesystem**
  - `train-tracking-server.py`: MLFlow python scripst to log metrics, models, parameters & register models using **``sqlite`` as tracking server**
- **``mlflow.db``**: Database file for sqlite
- **``README.md``**: Provides a setup guide for running the scripts locally.
- **``requirements.txt``**: Lists the dependencies required for the scripts.


## **.github/workflows Folder**
Contains GitHub Actions scripts, which are part of the solution for Question 2.
- it only runs `src/train-local-filesystem.py` because I didn't host any tracking server
- to runs `src/train-tracking-server.py` will need to set up storage server like Amazon S3 Bucket or Alibaba Cloud OSS
- The workflow includes the `git commit` and `git push` commands, but they have been commented out because there are no actual changes to the dataset.
    - As a result, Git detects that the working tree is clean and no changes need to be pushed, which triggers an error.
    - **To prevent this, the code for the pipeline has been commented out to ensure it ends successfully.**

## **Simulate-Remote-Storage Folder**
This folder simulates remote storage for Data Version Control (DVC) dataset versioning purposes.

