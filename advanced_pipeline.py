{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c09714c5",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2025-11-28T15:03:19.891728Z",
     "iopub.status.busy": "2025-11-28T15:03:19.891299Z",
     "iopub.status.idle": "2025-11-28T15:03:22.071019Z",
     "shell.execute_reply": "2025-11-28T15:03:22.069290Z"
    },
    "papermill": {
     "duration": 2.185172,
     "end_time": "2025-11-28T15:03:22.072944",
     "exception": false,
     "start_time": "2025-11-28T15:03:19.887772",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/input/hull-tactical-market-prediction/train.csv\n",
      "/kaggle/input/hull-tactical-market-prediction/test.csv\n",
      "/kaggle/input/hull-tactical-market-prediction/kaggle_evaluation/default_inference_server.py\n",
      "/kaggle/input/hull-tactical-market-prediction/kaggle_evaluation/default_gateway.py\n",
      "/kaggle/input/hull-tactical-market-prediction/kaggle_evaluation/__init__.py\n",
      "/kaggle/input/hull-tactical-market-prediction/kaggle_evaluation/core/templates.py\n",
      "/kaggle/input/hull-tactical-market-prediction/kaggle_evaluation/core/base_gateway.py\n",
      "/kaggle/input/hull-tactical-market-prediction/kaggle_evaluation/core/relay.py\n",
      "/kaggle/input/hull-tactical-market-prediction/kaggle_evaluation/core/kaggle_evaluation.proto\n",
      "/kaggle/input/hull-tactical-market-prediction/kaggle_evaluation/core/__init__.py\n",
      "/kaggle/input/hull-tactical-market-prediction/kaggle_evaluation/core/generated/kaggle_evaluation_pb2.py\n",
      "/kaggle/input/hull-tactical-market-prediction/kaggle_evaluation/core/generated/kaggle_evaluation_pb2_grpc.py\n",
      "/kaggle/input/hull-tactical-market-prediction/kaggle_evaluation/core/generated/__init__.py\n"
     ]
    }
   ],
   "source": [
    "# This Python 3 environment comes with many helpful analytics libraries installed\n",
    "# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python\n",
    "# For example, here's several helpful packages to load\n",
    "\n",
    "import numpy as np # linear algebra\n",
    "import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)\n",
    "\n",
    "# Input data files are available in the read-only \"../input/\" directory\n",
    "# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory\n",
    "\n",
    "import os\n",
    "for dirname, _, filenames in os.walk('/kaggle/input'):\n",
    "    for filename in filenames:\n",
    "        print(os.path.join(dirname, filename))\n",
    "\n",
    "# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using \"Save & Run All\" \n",
    "# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session"
   ]
  }
 ],
 "metadata": {
  "kaggle": {
   "accelerator": "none",
   "dataSources": [
    {
     "databundleVersionId": 14348714,
     "sourceId": 111543,
     "sourceType": "competition"
    }
   ],
   "isGpuEnabled": false,
   "isInternetEnabled": true,
   "language": "python",
   "sourceType": "notebook"
  },
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.13"
  },
  "papermill": {
   "default_parameters": {},
   "duration": 8.297086,
   "end_time": "2025-11-28T15:03:22.597996",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2025-11-28T15:03:14.300910",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
