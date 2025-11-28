{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "b809be1c",
   "metadata": {
    "_cell_guid": "b1076dfc-b9ad-4769-8c92-a6c4dae69d19",
    "_uuid": "8f2839f25d086af736a60e9eeb907d3b93b6e0e5",
    "execution": {
     "iopub.execute_input": "2025-11-28T15:01:31.031321Z",
     "iopub.status.busy": "2025-11-28T15:01:31.030422Z",
     "iopub.status.idle": "2025-11-28T15:01:33.045285Z",
     "shell.execute_reply": "2025-11-28T15:01:33.043811Z"
    },
    "papermill": {
     "duration": 2.019497,
     "end_time": "2025-11-28T15:01:33.046919",
     "exception": false,
     "start_time": "2025-11-28T15:01:31.027422",
     "status": "completed"
    },
    "tags": []
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/pytest.ini\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/advanced_pipeline.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/LICENSE\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/.gitignore\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/main.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/README.md\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/make_audit.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/requirements.txt\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/REPORT.md\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/notebookc10000092f\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/market_prediction_analysis.ipynb\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/make_submission.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/tests/conftest.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/tests/test_schema.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/tests/test_submission_format.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/tests/test_smoke.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/tests/test_advanced_pipeline.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/utils/metrics_logger.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/scripts/summarize_oof.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/.github/workflows/python.yml\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/kaggle_evaluation/default_inference_server.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/kaggle_evaluation/default_gateway.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/kaggle_evaluation/__init__.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/kaggle_evaluation/core/templates.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/kaggle_evaluation/core/base_gateway.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/kaggle_evaluation/core/relay.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/kaggle_evaluation/core/kaggle_evaluation.proto\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/kaggle_evaluation/core/__init__.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/kaggle_evaluation/core/generated/kaggle_evaluation_pb2.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/kaggle_evaluation/core/generated/kaggle_evaluation_pb2_grpc.py\n",
      "/kaggle/input/notebookc10000092f/-Hull-Tactical---Market-Prediction-main/kaggle_evaluation/core/generated/__init__.py\n"
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
     "datasetId": 8862810,
     "sourceId": 13909790,
     "sourceType": "datasetVersion"
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
   "duration": 7.355048,
   "end_time": "2025-11-28T15:01:33.568494",
   "environment_variables": {},
   "exception": null,
   "input_path": "__notebook__.ipynb",
   "output_path": "__notebook__.ipynb",
   "parameters": {},
   "start_time": "2025-11-28T15:01:26.213446",
   "version": "2.6.0"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
