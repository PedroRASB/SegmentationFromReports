"""
This code includes many functions to run the LLM on radiology or pathology reports.

inference_loop is the main function here. It will take as input the reports, call the LLM multiple times and write its outputs to a csv.
"""

import transformers
import torch
import os
import pandas as pd
import numpy as np
import math
import re
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
import random
from openai import OpenAI
import copy
from concurrent.futures import ThreadPoolExecutor
import csv
import ast
import time
import ast
from itertools import chain
import matplotlib.pyplot as plt
import tqdm
import httpx
import json

clt=None
mdl=None
def InitializeOpenAIClient(base_url='http://0.0.0.0:8000/v1'):
    global clt, mdl
    if clt is not None:
        return clt,mdl
    else:
        http_client = httpx.Client(
            trust_env=False,   # ignore HTTP_PROXY / HTTPS_PROXY, etc.
            verify=False       # no TLS cert check (local http endpoint)
        )
        # Initialize the client with the API key and base URL
        #clt = OpenAI(api_key='YOUR_API_KEY', base_url=base_url)
        clt = OpenAI(
            api_key="dummy",        # vLLM ignores the key
            base_url=base_url,      # e.g. "http://0.0.0.0:2328/v1"
            http_client=http_client # inject the transport
        )

        # Define the model name and the image path
        mdl = clt.models.list().data[0].id# Update this with the actual path to your PNG image
        print('Initialized model and client.')
        return clt,mdl

def CreateConversation(text, conver,role='user'):
    #if no previous conversation, send conver=[]. Do not automatically define conver above.
    cnv=copy.deepcopy(conver)
    
    cnv.append({
            'role': role,
            'content': [{
                'type': 'text',
                'text': text,
            }],
        })
    
    return cnv

def request_API(cv,model_name,client,max_tokens):
    print('Requesting API')

    if max_tokens is None:
        return client.chat.completions.create(
            model=model_name,
            messages=cv,
            temperature=0,
            top_p=1,
            timeout=6000)
    else:
        return client.chat.completions.create(
            model=model_name,
            messages=cv,
            max_tokens=max_tokens,
            temperature=0,
            top_p=1,
            timeout=6000)

def SendMessageAPI(text, conver, base_url='http://0.0.0.0:8000/v1',  
                    prt=True,max_tokens=None,
                    batch=1,
                    labels=None, id=None):
    """
    Sends a message to the LM deploy API.

    Args:
        text (str): The text message to send.
        conver (list): A list of conversation objects.
        base_url (str, optional): The base URL of the LM deploy API. Defaults to 'http://0.0.0.0:8000/v1'.
        size (int, optional): The size to resize the images to. Defaults to None.
        prt (bool, optional): Whether to print the images and conversation. Defaults to True.
        print_conversation (bool, optional): Whether to print the conversation. Defaults to False.
        max_tokens (int, optional): The maximum number of tokens in the completion response. Defaults to None.

    Returns:
        tuple: A tuple containing the updated conversation and the answer from the LM deploy API.
    """
    #if no previous conversation, send conver=[]. Do not automatically define conver above.
    client,model_name=InitializeOpenAIClient(base_url)

    if text is not None:
        if batch>1:
            for i in range(batch):
                #print('Batch:',i)
                #print('img_file_list:',img_file_list[i])
                #print('text:',text[i])
                #print('conver:',conver[i])
                conver[i]=CreateConversation(text=text[i], conver=conver[i])
        else:
            conver=CreateConversation(text=text, conver=conver)

    response=[]
    for i in range(batch):
        if batch==1:
            response=request_API(conver,model_name,client,max_tokens)
        else:
            # Use ThreadPoolExecutor to send both requests concurrently
            with ThreadPoolExecutor() as executor:
                # Map both the conversation and model name to each thread
                response = list(executor.map(request_API, conver, [model_name] * len(conver),[client] * len(conver),[max_tokens] * len(conver)))
        

    if batch==1:
        # Print the response
        answer = response.choices[0].message.content
        if prt:
            print('Conversation:')
            for item in conver:
                print(item['content'])
            if id is not None:
                print('ID:',id)
            print('Answer:',answer)
            if labels is not None:
                print('Labels:',labels)
        conver.append({"role": "assistant","content": [{"type": "text", "text": response.choices[0].message.content}]})
    else:
        answer=[]
        for i in range(batch):
            answer.append(response[i].choices[0].message.content)
            if prt:
                print('Conversation:')
                for item in conver[i]:
                    print(item['content'])
                
                if id is not None:
                    print('ID:',id[i])
                print('Answer:',answer[i])
                if labels is not None:
                    print('Labels:',labels[i])
            conver[i].append({"role": "assistant","content": [{"type": "text", "text": response[i].choices[0].message.content}]})

    return conver, answer











systemFastV0 = ("You are a knowledgeable, efficient, and direct AI assistant, and an expert in radiology reports."
    "Your answer should follow this template, "
    "substituting _ by 0 (indicating tumor absence), 1 (indicating tumor presence), or U (uncertain presence of tumor):"
    " liver tumor=_; kidney tumor=_; pancreas tumor=_")

system = ("You are a knowledgeable, efficient, and direct AI assistant, and an expert in radiology and radiology reports.")

instructions0ShotFastV0=("Instructions: Discover if a CT scan radiology report indicates the presence "
                   "of liver tumors, pancreas tumors or kidney tumors. Output binary labels for "
                   "each of these categories, where 1 indicates tumor presence, 0 tumor absence, and U uncertain presence of tumor. "
                   "Example: liver tumor presence=1; kidney tumor presence=U; pancreas tumor presence=0. "
                   "Answer with only the labels, do not repeat this prompt. ")

instructions0ShotFastLiverV1="""Instructions: Analyze a radiology report and answer the following questions. I want you to provide answers by filling a template, and not answering anything beyond this template.
1- Is any liver tumor present? 
Template--substitute _ by yes, no or uncertain: liver lesion presence=_;
Consider that: (a) 'unremarkable' means that an organ has no tumor; (b) organs not mentioned in the report have no tumor; (c) tumors may be described with many words, such as metastasis, tumor, lesion, mass, cyst, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic finding; (d) consider any lesion, hyperdensity or hypodensity a tumor, unless the report explicitly says that it is something else. Examples of lesions that are not tumors: ulcers, wounds, infections, inflammations, postinflammatory calcification, scars, renal calculi, nephrolithiasis, renal stones, or other diseases that are not tumors.
(e) Uncertany: You should answer uncertain if all findings if the report mentions abnormalities in the liver (e.g., hyperdensities or hypodensities), but says that they could not be characterized. Common words for uncertainty are: ill-defined, too small to characterize, and uncertain.
2- Is any of these types of liver tumor explicitly mentioned as present: Hepatic Hemangioma (HH), Focal Nodular Hyperplasia (FNH), Bile Duct Adenoma, Simple Liver Cyst (SLC), Hepatocellular Carcinoma (HCC), Cholangiocarcinoma (CCA), Hepatic Adenoma (HA), Mucinous Cystic Neoplasm (MCN)?
Template--substitute _ by yes, no, or uncertain (meaning the report explicitly indicates suspition for this type of lesion): HH=_; FNH=_; Bile Duct Adenoma=_; SLC=_; HCC=_; CCA=_; HA=_; MCN=_;
3- Is any malignant liver lesion present?
Template--substitute _ by yes, no or uncertain: malignant liver lesion presence=_;
Consider that: tumors are malignant if the report explictly mentions it, if it is growing (or reducing with cancer tratement), if it is cancer, or if it is an index lesion in an oncologic patient. HCC, CCA or matastasis are always malignant, HA, MCN are sometimes malignant, and HH, FNH, Bile Duct Adenoma, and SLC are benign. Remember that a patient may have both benign and malignant tumors.
4- What is the size of the largest malignant liver lesion in mm? 
Template: the template depends if the largest lesion is reported in 1D measurements (e.g., 15 mm), 2D measurements (e.g., 15 x 10 mm) or 3D measurements (e.g., 40 x 30 x 30 mm). ou may need to convert from cm to mm.
1D Template--substitute _ by the correct number (which should be 0 if there is no malignant liver tumor): largest malignant liver lesion size=_ mm;
2D Template--substitute _ by the correct number: largest malignant liver lesion size=_ x _ mm;
3D Template--substitute _ by the correct number: largest malignant liver lesion size=_ x _ x _ mm;
Consider that: (a) you may need to convert form cm to mm; (b) you must pay attention on which measurement refers to which lesion; (c) if tumor sizes are not informed for any malignant liver tumor, write "largest malignant liver lesion size=uncertain mm; (d) you should consider that the largest measured malignant lesion is the largest the patient has, unless the report explicitly says otherwise.
5- How many malignant tumors are present in the liver, if any?
Template--substitute _ by the correct number: number of malignant liver lesions=_
Consider that: (a) the number should be 0 if there is no malignant liver tumor; (b) write 'number of malignant lesions=uncertain' if the report mentions multiple lesions but does not count them.
6- How many tumors (benign and malignant) are present in the liver, if any?
Template--substitute _ by the correct number: number of liver lesions=_
Consider that: (a) the number should be 0 if there is no liver tumor; (b) write 'number of liver lesions=uncertain' if the report mentions multiple lesions but does not count them.
"""

instructions0ShotFastV2="""Instructions: Analyze a radiology report and answer the following questions. I want you to provide answers by filling a template, and not answering anything beyond this template.
1- Is any liver tumor present? 
Template--substitute _ by yes, no or uncertain: liver lesion presence=_;
Consider that: (a) 'unremarkable' means that an organ has no tumor; (b) organs not mentioned in the report have no tumor; (c) tumors may be described with many words, such as metastasis, tumor, lesion, mass, cyst, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic finding; (d) consider any lesion, hyperdensity or hypodensity a tumor, unless the report explicitly says that it is something else. Examples of lesions that are not tumors: ulcers, wounds, infections, inflammations, scars, renal calculi, nephrolithiasis, renal stones, or other diseases that are not tumors.
(e) Uncertainty: You should answer uncertain if the report mentions abnormalities in the liver (e.g., hyperdensities or hypodensities), but says that they could not be characterized. Common words for uncertainty are: ill-defined, too small to characterize, and uncertain.
If you answered no, skip the next liver questions (2-7) and go to the pancreas questions (do not include skipped questions in the template).
2- Is any of these types of liver tumor explicitly mentioned as present: Hepatic Hemangioma (HH), Focal Nodular Hyperplasia (FNH), Bile Duct Adenoma, Simple Liver Cyst (SLC), Hepatocellular Carcinoma (HCC), Cholangiocarcinoma (CCA), Hepatic Adenoma (HA), Mucinous Cystic Neoplasm (MCN)?
Template--substitute _ by yes, no, or uncertain (meaning the report explicitly indicates suspicion for this type of lesion): HH=_; FNH=_; Bile Duct Adenoma=_; SLC=_; HCC=_; CCA=_; HA=_; MCN=_;
3- Is any malignant liver lesion present?
Template--substitute _ by yes, no or uncertain: malignant liver lesion presence=_;
Consider that: tumors are malignant if the report explicitly mentions it, if it is growing (or reducing with cancer treatment), if it is cancer, or if it is an index lesion in an oncologic patient. HCC, CCA or metastasis are always malignant, HA, MCN are sometimes malignant, and HH, FNH, Bile Duct Adenoma, and SLC are benign. Remember that a patient may have both benign and malignant tumors.
4- What is the size of the largest malignant liver lesion, if any? 
Template: the template depends if the largest lesion is reported in 1D measurements (e.g., 15 mm), 2D measurements (e.g., 15 x 10 mm) or 3D measurements (e.g., 40 x 30 x 30 mm). You may write in cm or mm, as written in the report.
1D Template--substitute _ by the correct number (which should be 0 if there is no malignant liver tumor) and substitute unit by cm or mm: largest malignant liver lesion size=_ unit;
2D Template--substitute _ by the correct number and substitute unit by cm or mm: largest malignant liver lesion size=_ x _ unit;
3D Template--substitute _ by the correct number and substitute unit by cm or mm: largest malignant liver lesion size=_ x _ x _ unit;
Consider that: (a) you must pay attention to which measurement refers to which lesion; (b) you must check if the measurement if current or previous; (c) if tumor sizes are not informed for any malignant liver tumor, write "largest malignant liver lesion size=uncertain"; (d) you should consider that the largest measured malignant lesion is the largest the patient has, unless the report explicitly says otherwise.
5- What is the previous size of the lesion you mentioned in the last question, if any? 
Template--use the same structure as before (answer uncertain if this is not informed): previous size of the liver malignant lesion=_ unit; _ x _ unit; _ x _ x _ unit;
Consider that: Many reports have references to previous tumor sizes. For the largest current lesion, give me its previous size, if the report informs it. Pay close attention to past tenses, and adverbs like "previously" or "before", or references to dates. Analyze the syntax of the sentence to understand if the sizes are current or previous.
6- What is the size of the largest benign liver lesion, if any? 
Template--use the same structure as above: largest malignant liver lesion size=_ unit; _ x _ unit; _ x _ x _ unit;
7- How many malignant tumors are present in the liver, if any?
Template--substitute _ by the correct number: number of malignant liver lesions=_
Consider that: (a) the number should be 0 if there is no malignant liver tumor; (b) write 'number of malignant lesions=uncertain' if the report mentions multiple lesions but does not count them.
8- How many tumors (benign and malignant) are present in the liver, if any?
Template--substitute _ by the correct number: number of liver lesions=_
Consider that: (a) the number should be 0 if there is no liver tumor; (b) write 'number of liver lesions=uncertain' if the report mentions multiple lesions but does not count them.


### Pancreas Questions:
1- Is any pancreatic tumor present?
Template--substitute _ by yes, no or uncertain: pancreatic lesion presence=_;
Consider that: apply the same rules for identifying lesions in the pancreas. Use the list of tumors: Serous Cystadenoma (SCA), Mucinous Cystadenoma (MCA), Intraductal Papillary Mucinous Neoplasm (IPMN), Solid Pseudopapillary Neoplasm (SPN), Pancreatic Neuroendocrine Tumor (PNET), Pancreatic Ductal Adenocarcinoma (PDAC), Mucinous Cystadenocarcinoma (MCC).
If you answered no, skip the next pancreas questions (2-7) and go to the kidney questions (do not include skipped questions in the template).
2- Is any of these types of pancreatic tumors explicitly mentioned as present: Serous Cystadenoma (SCA), Mucinous Cystadenoma (MCA), Intraductal Papillary Mucinous Neoplasm (IPMN), Solid Pseudopapillary Neoplasm (SPN), Pancreatic Neuroendocrine Tumor (PNET), Pancreatic Ductal Adenocarcinoma (PDAC), Mucinous Cystadenocarcinoma (MCC)?
Template--substitute _ by yes, no, or uncertain: SCA=_; MCA=_; IPMN=_; SPN=_; PNET=_; PDAC=_; MCC=_;
3- Is any malignant pancreatic lesion present?
Template--substitute _ by yes, no or uncertain: malignant pancreatic lesion presence=_;
Remember: PDAC, MCC are always malignant, MCA, IPMN, SPN, and PNET may be malignant, and SCA is benign.
4- What is the size of the largest malignant pancreatic lesion? 
Template--use the same structure as for liver tumors: largest malignant pancreatic lesion size=_ unit; _ x _ unit; _ x _ x _ unit;
5- What is the previous size of the lesion you mentioned in the last question, if any? Many reports have references to previous tumor sizes. For the largest current lesion, give me its previous size, if the report informs it.
Template--use the same structure as before (answer uncertain if this is not informed): previous size of the malignant pancreas lesion=_ unit; _ x _ unit; _ x _ x _ unit;
6- What is the size of the largest benign pancreatic lesiont?
Template--use the same structure as for malignant lesions: largest benign pancreatic lesion size=_ unit; _ x _ unit; _ x _ x _ unit;
7- How many malignant tumors are present in the pancreas?
Template--substitute _ by the correct number: number of malignant pancreatic lesions=_
8- How many tumors (benign and malignant) are present in the pancreas?
Template--substitute _ by the correct number: number of pancreatic lesions=_

### Kidney Questions:
1- Is any kidney tumor present?
Template--substitute _ by yes, no or uncertain: kidney lesion presence=_;
Use the list of kidney tumors: Renal Oncocytoma (RO), Angiomyolipoma (AML), Simple Renal Cyst, Renal Cell Carcinoma (RCC), Transitional Cell Carcinoma (TCC), Wilms Tumor, Cystic Nephroma (CN), Multilocular Cystic Renal Neoplasm of Low Malignant Potential (MCRNLMP).
If you answered no, skip the next liver questions (2-7) and stop answering here  (do not include skipped questions in the template).
2- Is any of these types of kidney tumors explicitly mentioned as present: Renal Oncocytoma (RO), Angiomyolipoma (AML), Simple Renal Cyst, Renal Cell Carcinoma (RCC), Transitional Cell Carcinoma (TCC), Wilms Tumor, Cystic Nephroma (CN), Multilocular Cystic Renal Neoplasm of Low Malignant Potential (MCRNLMP)?
Template--substitute _ by yes, no, or uncertain: RO=_; AML=_; Simple Renal Cyst=_; RCC=_; TCC=_; Wilms Tumor=_; CN=_; MCRNLMP=_;
3- Is any malignant kidney lesion present?
Template--substitute _ by yes, no or uncertain: malignant kidney lesion presence=_;
Remember: RCC, TCC, Wilms Tumor are always malignant. RO, AML, Simple Renal Cyst are benign, and CN, MCRNLMP may be malignant.
4- What is the size of the largest malignant kidney lesion? 
Template--use the same structure as for liver tumors: largest malignant kidney lesion size=_ unit; _ x _ unit; _ x _ x _ unit;
5- What is the previous size of the lesion you mentioned in the last question, if any? Many reports have references to previous tumor sizes. For the largest current lesion, give me its previous size, if the report informs it.
Template--use the same structure as before (answer uncertain if this is not informed): previous size of the malignant kidney lesion=_ unit; _ x _ unit; _ x _ x _ unit;
6- What is the size of the largest benign kidney lesion?
Template--use the same structure as for malignant lesions: largest benign kidney lesion size=_ unit; _ x _ unit; _ x _ x _ unit;
7- How many malignant tumors are present in the kidneys?
Template--substitute _ by the correct number: number of malignant kidney lesions=_
8- How many tumors (benign and malignant) are present in the kidneys?
Template--substitute _ by the correct number: number of kidney lesions=_
"""


instructions0ShotFastCompact="""Instructions: Analyze a radiology report and answer the following questions for the liver, pancreas, and kidneys. I want you to provide answers by filling the template provided below for each organ (liver, pancreas, and kidneys), without answering anything beyond the template.
1- Is any tumor present in the liver, pancreas, or kidneys? 
Template--substitute _ by yes, no, or uncertain for each organ:
liver lesion presence=_; pancreatic lesion presence=_; kidney lesion presence=_;
Consider that: (a) 'unremarkable' means that an organ has no tumor; (b) organs not mentioned in the report have no tumor; (c) tumors may be described with many words, such as metastasis, tumor, lesion, mass, cyst, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic findings; (d) consider any lesion, hyperdensity, or hypodensity a tumor unless the report explicitly says it is something else (e.g., ulcers, infections, scars, renal calculi, nephrolithiasis, or other diseases).
(e) Uncertainty: Answer "uncertain" if abnormalities are present but could not be characterized (e.g., ill-defined, too small to characterize, or uncertain).
Stopping: If you answered no for all organs, stop answering here.

2- Are any of these specific types of tumors explicitly mentioned as present in the liver, pancreas, or kidneys? 
Template--substitute _ by yes, no, or uncertain for each tumor type:
Liver: HH=_; FNH=_; Bile Duct Adenoma=_; SLC=_; HCC=_; CCA=_; HA=_; MCN=_;  
Pancreas: SCA=_; MCA=_; IPMN=_; SPN=_; PNET=_; PDAC=_; MCC=_;  
Kidneys: RO=_; AML=_; Simple Renal Cyst=_; RCC=_; TCC=_; Wilms Tumor=_; CN=_; MCRNLMP=_;  

3- Is any malignant lesion present in the liver, pancreas, or kidneys?
Template--substitute _ by yes, no, or uncertain for each organ:
malignant liver lesion presence=_; malignant pancreatic lesion presence=_; malignant kidney lesion presence=_;
Consider that: (a) tumors are malignant if the report explicitly mentions it, if it is growing (or shrinking with cancer treatment), or if it is an index lesion in an oncologic patient; (b) specific tumors are always malignant (e.g., HCC, CCA, PDAC, MCC, RCC, TCC, Wilms Tumor, metastasis); (c) some tumors may be malignant (e.g., HA, MCN, MCA, IPMN, SPN, PNET, CN, MCRNLMP); and (d) benign tumors include HH, FNH, Bile Duct Adenoma, SLC, SCA, RO, AML, Simple Renal Cyst.

4- What is the size of the largest malignant lesion in each organ, in mm? 
Template--use the appropriate template based on whether the measurement is 1D, 2D, or 3D for each organ:
liver: largest malignant liver lesion size=_ mm / _ x _ mm / _ x _ x _ mm;
pancreas: largest malignant pancreatic lesion size=_ mm / _ x _ mm / _ x _ x _ mm;
kidneys: largest malignant kidney lesion size=_ mm / _ x _ mm / _ x _ x _ mm;
Consider that: (a) you may need to convert from cm to mm; (b) pay attention to which measurement refers to which lesion; (c) if sizes are not informed, write "uncertain"; (c) important: many reports have references to previous tumor sizes, you MUST ignore any previous measurements and consider only the largest lesion currently.

5- How many malignant tumors are present in the liver, pancreas, and kidneys?
Template--substitute _ by the correct number or uncertain for each organ:
number of malignant liver lesions=_; number of malignant pancreatic lesions=_; number of malignant kidney lesions=_;

6- How many tumors (benign and malignant) are present in the liver, pancreas, and kidneys?
Template--substitute _ by the correct number or uncertain for each organ:
number of liver lesions=_; number of pancreatic lesions=_; number of kidney lesions=_.
"""



instructions0Shot="""Carefully analyze the radiology report below, looking carefully at the findings and impressions sections (if available). Your task is answering the following questions:
1- Does the report indicate the presence of a liver tumor? Answer yes, no or it is uncertain. Justify your answer.
2- Does the report indicate the presence of a pancreas tumor? Answer yes, no or it is uncertain. Justify your answer.
3- Does the report indicate the presence of a kidney tumor? Answer yes, no or it is uncertain. Justify your answer.
After answering each of the 3 quesitons, fill in the following template, substituting _ by 'yes', 'no' or 'uncertain'. Do not change the template structure (e.g., keep using ; to separate the answers):
liver tumor presence=_; kidney tumor presence=_; pancreas tumor presence=_"""



instructions0ShotFast=("Instructions: Discover if the CT scan radiology report below indicates the presence "
                   "of liver tumors, pancreas tumors or kidney tumors. "
                   "Output labels for each of these categories, yes should indicate tumor presence, no tumor absence, and U uncertain presence of tumor. "
                   "Example: liver tumor presence=yes; kidney tumor presence=U; pancreas tumor presence=no. "
                   "Answer with only the labels, do not repeat this prompt. "
                   " Follow these rules for interpreting radiology reports: \n "
              "1- 'unremarkable' means that an organ has no tumor. \n "
              "2- Multiple words can be used to describe tumors, and you may check both the findings and impressions sections of the report (if present) to understand if an organ has tumors. Some words are: as metastasis, tumor, lesion, mass, cyst, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic finding"
              "3- Consider any lesion, hyperdensity or hypodensity a tumor, unless the report explicitly says that it is something else. "
              "Many conditions are not tumors, and should not be interpreted as so, unless a tumor is also reported along with the diease. Examples of liver conditions that are not tumors: Hepatitis, Cirrhosis, Fatty Liver Disease (FLD), Liver Fibrosis, Hemochromatosis, Primary Biliary Cholangitis (PBC), Primary Sclerosing Cholangitis (PSC), Wilson's Disease, Liver Abscess, Alpha-1 Antitrypsin Deficiency (A1ATD), steatosis, granulomas, Cholestasis, Budd-Chiari Syndrome (BCS), transplant, Gilbert's Syndromeulcers, wounds, infections, inflammations, and scars."
              "For the kidneys, some common conditions that are not tumors are: stents, inflammation, postinflammatory calcification, transplant, Chronic Kidney Disease (CKD), Acute Kidney Injury (AKI), Glomerulonephritis, Nephrotic Syndrome, Polycystic Kidney Disease (PKD), Pyelonephritis, Hydronephrosis, Renal Artery Stenosis (RAS), Diabetic Nephropathy, Hypertensive Nephrosclerosis, Interstitial Nephritis, Renal Tubular Acidosis (RTA), Goodpasture Syndrome, and Alport Syndrome. "
              "For the pancreas: Pancreatitis, Pancreatic Insufficiency, Cystic Fibrosis (CF), Diabetes Mellitus (DM), Exocrine Pancreatic Insufficiency (EPI), pancreatectomy, and Pancreatic Pseudocyst. \n"
              "Now some exmples of specific tumor names are: "
              "Liver: Hepatic Hemangioma (HH), Focal Nodular Hyperplasia (FNH), Bile Duct Adenoma, Simple Liver Cyst (SLC), Hepatocellular Carcinoma (HCC), Cholangiocarcinoma (CCA), Hepatic Adenoma (HA), Mucinous Cystic Neoplasm (MCN). \n "
                "Pancreas: Serous Cystadenoma (SCA), Pancreatic Ductal Adenocarcinoma (PDAC), Mucinous Cystadenocarcinoma (MCC), Mucinous Cystadenoma (MCA), Intraductal Papillary Mucinous Neoplasm (IPMN), Solid Pseudopapillary Neoplasm (SPN), Pancreatic Neuroendocrine Tumor (PNET). \n "
                "Kidney: Renal Oncocytoma (RO), Angiomyolipoma (AML), Simple Renal Cyst, Bosniak IIF cystic lesion, Renal Cell Carcinoma (RCC), Transitional Cell Carcinoma (TCC), Wilms Tumor, Cystic Nephroma (CN), Multilocular Cystic Renal Neoplasm of Low Malignant Potential (MCRNLMP), hydronephrosis, allograft. \n "
              "4- Consider any benign (e.g., cyst) or malingnat tumor a tumor. Thus, any type of cyst is a tumor. \n "
              "5- Organs never mentioned in the report have no tumors. \n "
                "6- Do not assume a lesion is uncertain unless it is explictly reported as uncertain. Many words can be used to describe uncertain lesions, such as: ill-defined, too small to characterize, nonspecific, and uncertain. Reports may express uncertainty about the tumor type (e.g., cyst or hemangioma), but certainty it is a tumor--in this case, you must consider the lesion a tumor. \n "
                "7- Organs with no tumor but other pathologies should be reported as 0.")
#100% accuracy!! but quite a few nans

instructionsLongitudinalPancreasAll=("Instructions: Below, I am providing you a sequence of radiology reports for CT scans of the same patient. They are numbered, as report 1, report 2, and so on. "
                         "Your task is to carefully read the reports, looking for pancreatic tumors (benign, malignant or cysts). "
                         "First, discover what is the first report that mentions a pancreatic tumor, if any. "
                         "Second, carefully read the reports before the first report that mentions a pancreatic tumor. From these reports, which ones do not mention any suspicious findings in the pancreas? Those are pre-diagnosis reports. \n"
                         "In your answer, fill out the following template: \n"
                         "first diagnosis report=_; pre-diagnosis reports=_;\n"
                         "An example is: first diagnosis report=3; pre-diagnosis reports=1,2;\n"
                         "If you find no pancreatic tumor in any report, answer: \n"
                         "An example is: first diagnosis report=none; pre-diagnosis reports=none;\n"
                         "If you find no pancreatic pre-diagnostic report (i.e., report 1 shows sign of a pancreatic tumor), answer pre-diagnosis reports=none\n"
                  " Follow these rules for interpreting radiology reports: \n "
             "1- 'unremarkable' means that an organ has no tumor. \n "
             "2- Multiple words can be used to describe tumors, and you may check both the findings and impressions sections of the report (if present) to understand if an organ has tumors. Some words are: as metastasis, tumor, lesion, mass, cyst, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic finding"
             "3- Consider any lesion, hyperdensity or hypodensity a tumor, unless the report explicitly says that it is something else. "
             "Many conditions are not tumors, and should not be interpreted as so, unless a tumor is also reported along with the disease. "
             "Example of conditions that are not tumors: Pancreatitis, Pancreatic Insufficiency, Cystic Fibrosis (CF), Diabetes Mellitus (DM), Exocrine Pancreatic Insufficiency (EPI), pancreatectomy, Pancreatic Pseudocyst, and fat infiltration. \n"
             "Now some examples of specific tumor names are: Serous Cystadenoma (SCA), Pancreatic Ductal Adenocarcinoma (PDAC), Mucinous Cystadenocarcinoma (MCC), Mucinous Cystadenoma (MCA), Intraductal Papillary Mucinous Neoplasm (IPMN), Solid Pseudopapillary Neoplasm (SPN), Pancreatic Neuroendocrine Tumor (PNET). \n "
             "4- Consider any benign (e.g., cyst) or malignant tumor a tumor. Thus, any type of cyst is a tumor. \n "
             "5- If the pancreas is not mentioned in a report, the report indicates no pancreatic tumors.\n"
             "6- No report AFTER the first diagnosis can be considered pre-diagnostic, even if they show no pancreatic tumor. Pre-diagnostic reports are only those before the first report showing a pancreas tumor. \n"
             "7- If no report mentions a pancreatic tumor, YOU MUST ANSWER 'none' for both first diagnosis and pre-diagnosis reports. \n"
             "8- If a report mentions a Whipple procedure or any pancreatic surgery, you CANNOT consider it a pre-diagnostic report. \n"
             "Justify your answer, explaining why the first diagnosis report has a pancreatic tumor, and why the pre-diagnosis reports do not have suspicion of pancreatic tumors. Cite sentences from the reports to support your justification.")


instructionsLongitudinalPancreas=("Instructions: Below, I am providing you a sequence of radiology reports for CT scans of the same patient. They are numbered, as report 1, report 2, and so on. "
                         "Your task is to carefully read the reports, looking for pancreatic tumors. "
                         "First, discover what is the first report that mentions a pancreatic tumor, if any. "
                         "Second, carefully read the reports before the first report that mentions a pancreatic tumor. Among these reports, which ones do not mention any suspicious findings in the pancreas? Those are pre-diagnosis reports. \n"
                         "In your answer, fill out the following template: \n"
                         "first diagnosis report=_; pre-diagnosis reports=_;\n"
                         "An example is: first diagnosis report=3; pre-diagnosis reports=1,2;\n"
                         "If you find no pancreatic tumor in any report, answer: \n"
                         "An example is: first diagnosis report=none; pre-diagnosis reports=none;\n"
                         "If you find no pancreatic pre-diagnostic report (i.e., report 1 shows sign of a pancreatic tumor), answer pre-diagnosis reports=none\n"
                  "Closely follow these rules, pay great attention to them: \n "
             "1- 'unremarkable' means that an organ has no tumor. If the pancreas is not mentioned in a report, the report indicates no pancreatic tumors. \n "
             "2- Multiple words can be used to describe tumors, and you may check both the findings and impressions sections of the report (if present) to understand if an organ has tumors. Some words are: as tumor, lesion, mass, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic finding"
             "3- Consider any lesion, hyperdensity or hypodensity a tumor, unless the report explicitly says that it is something else. "
             "Many conditions are not tumors, and should not be interpreted as so, unless a tumor is also reported along with the disease. "
             "Example of conditions that are not tumors: Pancreatitis, Pancreatic Insufficiency, Cystic Fibrosis (CF), Diabetes Mellitus (DM), Exocrine Pancreatic Insufficiency (EPI), pancreatectomy, Pancreatic Pseudocyst, and fat infiltration. \n"
             "4- No report AFTER the first diagnosis can be considered pre-diagnostic, even if they show no pancreatic tumor. Pre-diagnostic reports are only those before the first report showing a pancreas tumor. \n"
             "5- If no report mentions a pancreatic tumor, YOU MUST ANSWER 'none' for both first diagnosis and pre-diagnosis reports. \n"
             "6- If a report mentions a Whipple procedure, pancreatectomy, or any pancreatic surgery, you CANNOT consider it a pre-diagnostic report. \n"
             "7- If a report mentions pancreatic duct dilation, suspicion of a pancreatic tumor or focal pancreatic atrophy, you CANNOT consider it a pre-diagnostic report. \n\n"
             "Justify your answer, explaining why the first diagnosis report has a pancreatic tumor, and why the pre-diagnosis reports do not have suspicion of pancreatic tumors. Cite sentences from the reports to support your justification.")



instructionsLongitudinalPancreasDiagnosis=("Instructions: Below, I am providing you a sequence of radiology reports for CT scans of the same patient. They are numbered, as report 1, report 2, and so on. "
                         "Your task is to carefully read the reports, looking for pancreatic tumors, and discover the types of pancreatic tumors the patient had. "
                         "Use the tumor locations (pancreas head, body and tail) to track the tumors through time. This information can help you discover if a tumor type was unknown in a previous report, but clarified in a later one. \n"
                         "You must list all types of pancreatic tumors the patient had in the reports. \n"
                         "Answer by filling out the following template: \n"
                         "tumor types: _; _; _; \n"
                         "You should list all tumor types you find, separating them by semicolons. You will not necessairly find 3 types, ajust the number of semicolons to the number of tumor types you find. \n"
                         "Example: \n"
                         "tumor types: PDAC; Cyst; Unknown; \n"
                         "Only say Unknown if a the type of one or more tumors are not mentioned in any of the reports. \n"
                         "An example is: first diagnosis report=3; pre-diagnosis reports=1,2;\n"
                         "If you find no pancreatic tumor in any report, answer: \n"
                         "tumor types: none;\n"
                  "Closely follow these rules for interpreting report, pay great attention to them: \n "
             "1- In your answer, try using the abbreviations in the list below:"
             "Unknown, Cyst, Metastasis, Serous Cystadenoma (SCA), Pancreatic Ductal Adenocarcinoma (PDAC), Mucinous Cystadenocarcinoma (MCC), Mucinous Cystadenoma (MCA), Intraductal Papillary Mucinous Neoplasm (IPMN), Solid Pseudopapillary Neoplasm (SPN), Pancreatic Neuroendocrine Tumor (PNET)."
             "1- If a report says the pancreas is 'unremarkable', it means that it has no tumor. If the pancreas is not mentioned in a report, the report indicates no pancreatic tumors. \n "
             "2- Multiple words can be used to describe tumors, and you may check both the findings and impressions sections of the report (if present) to understand if an organ has tumors. Some words are: as tumor, lesion, mass, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic finding"
             "3- Consider any lesion, hyperdensity or hypodensity a tumor, unless the report explicitly says that it is something else. "
             "4- NEVER guess. If a tumor type is not explicitly mentioned in any report, you must say Unknown. \n"
             "5- Remember all reports belong to the same parient and appear in chronological order. Try your best to track the tumors through time, to discover their types if possible. \n"
             "Many conditions are not tumors, and should not be interpreted as so, unless a tumor is also reported along with the disease. "
             "Example of conditions that are not tumors: Pancreatitis, Pancreatic Insufficiency, Cystic Fibrosis (CF), Diabetes Mellitus (DM), Exocrine Pancreatic Insufficiency (EPI), pancreatectomy, Pancreatic Pseudocyst, and fat infiltration. \n"
             "ALWAYS CAREFULLY JUSTIFY your answer, explain from which sentences in the report you deduced the tumor type. Also, explain how do you trace each pancreatic tumor across diverse reports, and what were their types in each report. Explain which pancreatic tumors have unknown types (across all reports), if any. Cite sentences from the reports to support your justification.")


preDiagnositcConfirmation=("Instructions: Below, I am providing you a radiology report for a CT scan. "
                         "Your task is to carefully read the report and identify if it presents any tumor suspicion in the pancreas, a pancreas surgery, or history of cancer. "
                         "In your answer, fill out the following template: \n"
                         "pancreatic tumor suspicion=_; pancreas surgery=_; cancer history=_\n"
                         "Replace _ by yes or no. \n"
                  "Closely follow these rules, pay great attention to them: \n "
             "1- 'unremarkable' means that an organ has no tumor suspicion. \n" 
             "2- If the pancreas is not mentioned in a report, the report indicates no pancreatic tumor suspicion. \n "
             "3- Consider that mentions of pancreatic duct dilation, biliary duct dilation or focal pancreatic atrophy are tumor signs. In this case, answer pancreatic tumor suspicion=yes; ...\n"
             "4- Multiple words can be used to describe tumors, and you may check both the findings and impressions sections of the report (if present) to understand if the pancreas has tumor suspicions. Some words are: as tumor, lesion, mass, neoplasm, growth, cancer, index lesion in cancer patients, and lesions listed as oncologic finding."
             "Many conditions are not tumors, and should not be interpreted as so, unless a tumor is also reported along with the disease. Do not consider small hyper/hypo densities a tumor, unless the report shows suspicion of a pancreatic tumor. "
             "Example of conditions that are not tumors: Pancreatitis, Pancreatic Insufficiency, Cystic Fibrosis (CF), Diabetes Mellitus (DM), Exocrine Pancreatic Insufficiency (EPI), Pancreatic Pseudocyst, and fat infiltration. \n"
             "6- Examples of pancreas surgery are Whipple procedure, pancreatectomy and Pancreaticoduodenectomy. \n"
             "7- To fill out the history of cancer part, carefully check if the report mentions any cancer treatment of diagnosis. Check especially in the clinical history section (if present) \n"
             "Justify your answer explaining why the report shows suspicion of pancreatic tumors or not. Cite sentences from the reports to support your justification.")


instructions0ShotMalignancyFast=("Instructions: The radiology report below mentions a %(organ)s tumor (or tumors). Read it carefully, paying special attention to the findings, clinical history and impressions sections (if available). \n"
                     "Does the report mention any malignant tumor in the %(organ)s? \n"
                   "Answer me by just filling out the template below, substituting _ by yes (malignancy present), no (malignancy absence), or U (uncertain presence of malignancy): \n "
                   "malignant tumor in %(organ)s=_ \n"
                   "Answer with only the filled template, do not repeat this prompt. "
                   " Follow these rules for interpreting radiology reports: \n "
                    "1- Some words are only used for describing malignant tumors, for example: metastasis, cancer, growing, or any oncologic lesion and index lesion in cancer patients. \n"
                    "Reports may sometimes mention the specific tumor type. In the %(organ)s, benign tumors are: %(benign_tumors)s. \n "
                    "Malignant tumors are: %(malignant_tumors)s. \n "
                    "Tumors that may be both benign or malignant are: %(both_tumors)s. \n "
                    "2- If the report does not mention that the tumor is benign or does not specify lesion type, but the tumor is growing in relation to a past measurement (with no benign explanation--e.g., cysts can grow and are benign), consider it malignant. \n"
                    "3- If the report impressions explicitly state no abnormality in the %(organ)s or abdomen, assume no malignancy. \n"
                    "4- It the report does not mention the tumor type, but you read that the patient has cancer in the %(organ)s, or has history of malignant tumors in the %(organ)s (analyze the clinical history or finginds sections if they are present), consider the tumor malignant. \n")
#if they do not say it is malignant but it is growing, it is malignant, or it is a cancer patient or it is an index lesion and not specifically benign, it is malignant


instructions0ShotMalignancy=("Instructions: The radiology report below mentions a %(organ)s tumor (or tumors). Read it carefully, paying special attention to the findings, clinical history and impressions sections (if available). \n"
                     "Does the report mention any malignant tumor in the %(organ)s? \n"
                   "Answer me by just filling out the template below, substituting _ by yes (malignancy present), no (malignancy absence), or U (uncertain presence of malignancy): \n "
                   "malignant tumor in %(organ)s=_ \n"
                   "Besides filling the template, justify your answer, carefully mentioning each section of the report if present: clinical history, findings and impressions. "
                   " Follow these rules for interpreting radiology reports: \n "
                    "1- Some words are only used for describing malignant tumors, for example: metastasis, cancer, growing, or any oncologic lesion and index lesion in cancer patients. \n"
                    "Reports may sometimes mention the specific tumor type. In the %(organ)s, benign tumors are: %(benign_tumors)s. \n "
                    "Malignant tumors are: %(malignant_tumors)s. \n "
                    "Tumors that may be both benign or malignant are: %(both_tumors)s. \n "
                    "2- If the report does not mention that the tumor is benign or does not specify lesion type, but the tumor is growing in relation to a past measurement (with no benign explanation--e.g., cysts can grow and are benign), consider it malignant. \n"
                    "3- If the report impressions explicitly state no abnormality in the %(organ)s or abdomen, assume no malignancy. \n"
                    "4- It the report does not mention the tumor type, but you read that the patient has cancer in the %(organ)s, or has history of malignant tumors in the %(organ)s (analyze the clinical history or finginds sections if they are present), consider the tumor malignant. \n")

instructions0ShotMalignantSize=("Instructions: The radiology report below mentions a malignant tumor (or tumors) in the %(organ)s. "
                    "Read it carefully, paying special attention to the findings, clinical history and impressions sections (if available). \n"
                    "Your task is to list the sizes and locations of all malignant tumors in the %(organ)s. \n"
                    "Fill out the template below, using one line per malignant tumor in the %(organ)s (you may add or remove lines from the template). Substitute the first _ in each line by the the tumor size, and the second by its location: \n "
                    "%(organ)s malignant tumor size = _; location = _;\n"
                    "%(organ)s malignant tumor size = _; location = _;\n"
                    "... \n"
                    "Reports can write the size of the tumor in 1D, 2D or 3D measurements, and you should use the same standards used in the report."
                    "Write 1D measurements as: 15 mm; 2D measurements as: 15 x 10 mm; and 3D measurements as: 40 x 30 x 30 mm. You may use either cm or mm, but you MUST WRITE in each line of the filled template the unit you are using (cm or mm). If a report does not specify the unit, assume it is mm. \n"
                    "For location, chose one of these options for each tumor: %(organ_locations)s \n"
                    "You can use location=U if the report does not specify the location of the tumor. \n"
                    "Besides filling the template, justify your answer, carefully mentioning each section of the report if present: clinical history, findings and impressions.\n"
                    "Some report may refer to past measurements (using words like previously, before, or giving dates). Ignore previous measuremtns. Provide me a synthatic analysis of the report sentences mentioning %(organ)s tumor sizes. In this analysis, explain which measurement refers to which tumor, if the measurement is current or past, and if the corresponding tumor is malignant or benign.\n"
                    "Follow these rules for interpreting radiology reports: \n "
                    "1- Some words are only used for describing malignant tumors, for example: metastasis, cancer, growing, or any oncologic lesion and index lesion in cancer patients. \n"
                    "Reports may sometimes mention the specific tumor type. In the %(organ)s, benign tumors are: %(benign_tumors)s. \n "
                    "Malignant tumors are: %(malignant_tumors)s. \n "
                    "Tumors that may be both benign or malignant are: %(both_tumors)s. \n "
                    "2- If the report does not mention that the tumor is benign or does not specify lesion type, but the tumor is growing in relation to a past measurement (with no benign explanation--e.g., cysts can grow and are benign), consider it malignant. \n"
                    "3- If the report impressions explicitly state no abnormality in the %(organ)s or abdomen, assume no malignancy. \n"
                    "4- If the report does not mention the tumor type, but you read that the patient has cancer in the %(organ)s, or has history of malignant tumors in the %(organ)s (analyze the clinical history or finginds sections if they are present), consider the tumor malignant. \n"
                    "5- If the report mentions multiple malignant tumors in the %(organ)s, list the sizes of all of them. \n"
                    "6- If the report does not mention a certain tumor size, write 'U' to indicate uncertain (e.g., malignant tumor 2 = U). \n")

instructions0ShotSizenType = (
    "Instructions: The radiology report below mentions one or more tumors in the %(organ)s. "
    "Read it carefully, paying special attention to the findings, clinical history, and impressions sections (if available). \n"
    "Your task is to list the types, certainty of tumor type, sizes, and locations of all tumors in the %(organ)s. \n"
    "Fill out the template below, using one line per tumor in the %(organ)s (you may add or remove lines from the template): \n"
    "%(organ)s tumor 1: type = _; certainty = _; size = _; location = _;\n"
    "%(organ)s tumor 2: type = _; certainty = _; size = _; location = _;\n"
    "... \n"
    "Disregard tumors only if none of the following details are provided: size, location, or type. \n"
    "\nSize: "
    "Reports can write the size of the tumor in 1D, 2D, or 3D measurements, and you should use the same standards used in the report. "
    "Write 1D measurements as: 15 mm; 2D measurements as: 15 x 10 mm; and 3D measurements as: 40 x 30 x 30 mm. You may use either cm or mm, but you MUST WRITE in each line of the filled template the unit you are using (cm or mm). If a report does not specify the unit, assume it is mm. \n"
    "Say size = U if the report does not specify the size of a tumor. \n"
    "\nLocation: "
    "For location, choose one of these options for each tumor: %(organ_locations)s \n"
    "Say location = U if the report does not specify the location of a tumor. \n"
    "\nType: "
    "If the report informs tumor type, inform it. Tumor type list: %(benign_tumors)s, %(malignant_tumors)s, %(both_tumors)s. \n"
    "Otherwise, say type = U if the report does not specify the type of the tumor. \n"
    "Follow these rules:"
    "1- If the tumor type is not specified in findings, you may deduce it from the clinical history or impressions sections. \n"
    "2- Cyst, or cystic lesion, is a common type of tumor. For any cysts, you say type = cyst. \n"
    "3- Assign type = metastasis if the %(organ)s tumor is described as a metastasis originating from a cancer in another organ. If the report mentions metastatic cancer in **another organ** along with lesions in the %(organ)s, classify the %(organ)s lesions as type = metastasis unless explicitly stated otherwise. Determine certainty based on the level of confidence expressed in the report. \n"
    "4- If the report does not mention the tumor type, but you read that the patient has cancer in the %(organ)s, or has a history of malignant tumors in the %(organ)s, you may say type = cancer. However, do not say type = cancer if a more specific tumor type is given (like cyst, PDAC and PNET in pancreas, RCC in kidney, HCC in liver,...). \n"
    "5- Try using terms from the tumor type list. E.g., if the list says 'Pancreatic Ductal Adenocarcinoma (PDAC)' and the report mentions 'adenocarcinoma in the pancreas', say type = Pancreatic Ductal Adenocarcinoma (PDAC). \n"
    "%(extra_info)s"
    "\nCertainty: "
    "Certainty of the tumor type, according to the report. If a report mentions a tumor type in the findings, history, or impressions, without demonstrating uncertainty, say certainty = certain. "
    "If the report expresses strong confidence in tumor type, say certainty = high. "
    "If the report mentions a tumor type but expresses significant uncertainty about it, say certainty = low. "
    "If the report does not mention the tumor type, say certainty = U. \n"
    "\nJustification: "
    "Besides filling the template, justify your answer, carefully mentioning each section of the report if present: history, findings, and impressions. "
    "Explain from which sentences you got each size, location, and type.\n"
    "Some reports may refer to past measurements (using words like previously, before, or giving dates). Ignore previous measurements. "
    "Provide me a syntactic analysis of the report sentences mentioning %(organ)s tumor sizes. In this analysis, explain which measurement refers to which tumor, if the measurement is current or past, and if the corresponding tumor is malignant or benign.\n"
    "I will provide an example of a report and a correct answer for a %(organ)s tumor:\n"
    "Example report: \n"
    "%(example_report)s \n"
    "Example answer: \n"
    "%(example_answer)s \n"
    "\n"
    "End of the example. \n"
    "\n"
)


instructionsHCC = (
    "Instructions: The radiology report below may mention one or more tumors in the liver. "
    "Read it carefully, paying special attention to the findings, clinical history, and impressions sections (if available). \n"
    "Your task is to list the types, certainty of tumor type, sizes, locations, arterial enhancement, washout, capsule, threshold growth, and LI-RADS score of all tumors in the liver. \n"
    "Fill out the template below, using one line per tumor in the liver (you may add or remove lines from the template): \n"
    "liver tumor 1: type = _; certainty = _; size = _; location = _; arterial enhancement = _; washout = _; capsule = _; threshold growth = _; LI-RADS = _;\n"
    "liver tumor 2: type = _; certainty = _; size = _; location = _; arterial enhancement = _; washout = _; capsule = _; threshold growth = _; LI-RADS = _;\n"
    "... \n"
    "Disregard tumors only if none of the following details are provided: size, location, or type. \n"
    "\nSize: "
    "Reports can write the size of the tumor in 1D, 2D, or 3D measurements, and you should use the same standards used in the report. "
    "Write 1D measurements as: 15 mm; 2D measurements as: 15 x 10 mm; and 3D measurements as: 40 x 30 x 30 mm. You may use either cm or mm, but you MUST WRITE in each line of the filled template the unit you are using (cm or mm). If a report does not specify the unit, assume it is mm. \n"
    "Say size = U if the report does not specify the size of a tumor. \n"
    "\nLocation: "
    "For location, choose one of these options for each tumor: %(organ_locations)s \n"
    "Say location = U if the report does not specify the location of a tumor. \n"
    "\nType: "
    "If using the LI-RADS classification, the rpeort is evaluating for HCC, so HCC is a probable type. \n"
    "If the report informs tumor type, inform it. Tumor type list: %(benign_tumors)s, %(malignant_tumors)s, %(both_tumors)s. \n"
    "Otherwise, say type = U if the report does not specify the type of the tumor. \n"
    "Follow these rules:"
    "1- If the tumor type is not specified in findings, you may deduce it from the clinical history or impressions sections. \n"
    "2- Cyst, or cystic lesion, is a common type of tumor. For any cysts, you say type = cyst. \n"
    "3- Assign type = metastasis if the liver tumor is described as a metastasis originating from a cancer in another organ. If the report mentions metastatic cancer in **another organ** along with lesions in the liver, classify the liver lesions as type = metastasis unless explicitly stated otherwise. Determine certainty based on the level of confidence expressed in the report. \n"
    "4- If the report does not mention the tumor type, but you read that the patient has cancer in the liver, or has a history of malignant tumors in the liver, you may say type = cancer. However, do not say type = cancer if a more specific tumor type is given (like cyst, PDAC and PNET in pancreas, RCC in kidney, HCC in liver,...). \n"
    "5- Try using terms from the tumor type list. E.g., if the list says 'Pancreatic Ductal Adenocarcinoma (PDAC)' and the report mentions 'adenocarcinoma in the pancreas', say type = Pancreatic Ductal Adenocarcinoma (PDAC). \n"
    "%(extra_info)s"
    "\nCertainty: "
    "Certainty of the tumor type, according to the report. If a report mentions a tumor type in the findings, history, or impressions, without demonstrating uncertainty, say certainty = certain. "
    "If the report expresses strong confidence in tumor type, say certainty = high. "
    "If the report mentions a tumor type but expresses significant uncertainty about it, say certainty = low. "
    "If the report does not mention the tumor type, say certainty = U. \n"
    "\nArterial enhancement: "
    "Arterial phase hyperenhancement (APHE) is non-rim arterial hyperenhancement of a lesion which is greater than the enhancement of the surrounding liver. \n"
    "It is a main consideration in the LI-RADS classification, and should me mentioned explicitly. \n"
    "If the report does not mention the arterial enhancement, say arterial enhancement = U. \n"
    "\nWashout: "
    "Non-peripheral washout is a decrease in attenuation or intensity from earlier to later phase, resulting in hypoenhancement in the portal venous or delayed phase. \n"
    "It is a main consideration in the LI-RADS classification, and should me mentioned explicitly. \n"
    "If the report does not mention the washout, say washout = U. \n"
    "\nCapsule: "
    "Capsule is a smooth, uniform border surrounding all or most of an observation. \n"
    "It is a main consideration in the LI-RADS classification, and should me mentioned explicitly. \n"
    "If the report does not mention the capsule, say capsule = U. \n"
    "\nThreshold growth: "
    "Threshold growth is increase in lesion size of 50 or more within 6 months time during follow-up imaging. \n"
    "It is a main consideration in the LI-RADS classification, and should me mentioned explicitly. \n"
    "If the report does not mention the threshold growth, say threshold growth = U. \n"
    "\nLI-RADS score: "
    "LI-RADS score is a classification of the likelihood of malignancy of a HCC tumor, based on the size, location, and enhancement of the tumor. \n"
    "It should be mentioned explicitly. \n"
    "If the report does not mention the LI-RADS score, say LI-RADS = U. \n"
    "\nJustification: "
    "Besides filling the template, justify your answer, carefully mentioning each section of the report if present: history, findings, and impressions. "
    "Explain from which sentences you got each size, location, type, arterial enhancement, washout, capsule, threshold growth, and LI-RADS score.\n"
    "Some reports may refer to past measurements (using words like previously, before, or giving dates). Ignore previous measurements. "
    "Provide me a syntactic analysis of the report sentences mentioning liver tumor sizes, types, arterial enhancement, washout, capsule, threshold growth, and LI-RADS score. In this analysis, explain which measurement refers to which tumor, if the measurement is current or past, and if the corresponding tumor is malignant or benign.\n"
    "I will provide an example of a report and a correct answer for a liver tumor:\n"
    "Example report: \n"
    "%(example_report)s \n"
    "Example answer: \n"
    "%(example_answer)s \n"
    "\n"
    "End of the example. \n"
    "\n"
    "Now the report you should analyze: \n"
)

reportHCC = """
MR ABDOMEN WITH AND WITHOUT CONTRAST 4/1/2024 
COMPARISON:  MR abdomen 4/10/2023
CLINICAL HISTORY:  HBV and Cirrhosis and history of liver lesions, rule out HCC
TECHNIQUE:  MRI of the abdomen was performed.
MEDICATIONS:
Dotarem - 15 mL - Intravenous
FINDINGS:
Liver:  Nodular contour consistent with cirrhosis. Patent portal and hepatic veins. Observations as follows:
Untreated observation 
Location: Hepatic segment 2 
Size: 1.4 cm (Se/Im 902/16), 
Threshold growth: previously measured 0.7 cm on 4/10/2023
Arterial enhancement Pattern: Non-rim hyperenhancement  
Presence of Washout: Present
Enhancing Capsule: Present
Other findings: T2 hyperintensity
LI-RADS: 5
Additional scattered arterially hyperenhancing foci predominantly in the periphery of liver, without washout, collectively categorized as LI-RADS 3.
Gallbladder: Cholelithiasis without cholecystitis.
Spleen:  Splenomegaly, 13.7 cm.
Pancreas:  Unremarkable 
Adrenal Glands:  Unremarkable
Kidneys:  Right renal cyst.
GI Tract:  Unremarkable
Vasculature:  Distal paraesophageal varices
Lymphadenopathy: Absent
Peritoneum: No ascites
Bones:  No suspicious lesions

IMPRESSION: 
1.  Cirrhosis with 1.4 cm segment 2 LI-RADS 5 observation.
2.  Additional scattered arterially hyperenhancing foci without washout, collectively LI-RADS 3.
"""

answerHCC = """
liver tumor 1: type = HCC; certainty = certain; size = 1.4 cm; location = segment 2; arterial enhancement = present; washout = present; capsule = present; threshold growth = present; LI-RADS = 5;
liver tumor 2: type = U; certainty = low; size = U; location = U; arterial enhancement = present; washout = absent; capsule = U; threshold growth = U; LI-RADS = 3;

Justification:
- History: The patient has HBV, cirrhosis, and a history of liver lesions, which raises the likelihood of HCC.
- Findings:
  • A 1.4 cm lesion in segment 2, previously 0.7 cm → threshold growth present.
  • Arterial hyperenhancement (non-rim), washout, and a capsule are all HCC indicators.
  • LI-RADS = 5, confirming a definite HCC diagnosis, so type = HCC, certainty = certain.
  • Multiple scattered arterially hyperenhancing foci in the liver periphery are listed as LI-RADS 3 (indeterminate). No reported size or segment → size = U, location = U, type = U, certainty = low, arterial enhancement present, washout absent, capsule/threshold growth unmentioned (U).
  
Syntactic Analysis of Report:
1) “Untreated observation ... Size: 1.4 cm ... Threshold growth: previously 0.7 cm on 4/10/2023. Arterial enhancement Pattern: Non-rim hyperenhancement, Presence of Washout: Present, Enhancing Capsule: Present, LI-RADS: 5.”
   - This lesion is a classic HCC (all major LI-RADS features present).
2) “Additional scattered arterially hyperenhancing foci ... collectively LI-RADS 3.”
   - Indeterminate lesions (no washout, no size info) → type = U, certainty = low, LI-RADS = 3.
"""

instructions0ShotSizenTypePathology = (
    "Instructions: The pathology report below mentions one or more tumors in the %(organ)s. "
    "Read it carefully, paying special attention to the Final Diagnosis section, which is usually in the beginning of the report. \n"
    "Your task is to list the types, certainty of tumor type, sizes, and locations of all tumors in the %(organ)s. \n"
    "Fill out the template below, using one line per tumor in the %(organ)s (you may add or remove lines from the template): \n"
    "%(organ)s tumor 1: type = _; certainty = _; size = _; location = _;\n"
    "%(organ)s tumor 2: type = _; certainty = _; size = _; location = _;\n"
    "... \n"
    "Disregard tumors only if none of the following details are provided: size, location, or type. \n"
    "\nSize: "
    "Reports can write the size of the tumor in 1D, 2D, or 3D measurements, and you should use the same standards used in the report. "
    "Write 1D measurements as: 15 mm; 2D measurements as: 15 x 10 mm; and 3D measurements as: 40 x 30 x 30 mm. You may use either cm or mm, but you MUST WRITE in each line of the filled template the unit you are using (cm or mm). If a report does not specify the unit, assume it is mm. \n"
    "Say size = U if the report does not specify the size of a tumor. \n"
    "\nLocation: "
    "For location, choose one of these options for each tumor: %(organ_locations)s \n"
    "Say location = U if the report does not specify the location of a tumor. \n"
    "\nType: "
    "If the report informs tumor type, inform it. Tumor type list: %(benign_tumors)s, %(malignant_tumors)s, %(both_tumors)s. \n"
    "Otherwise, say type = U if the report does not specify the type of the tumor. \n"
    "Follow these rules:"
    "1- Cyst, or cystic lesion, is a common type of tumor. For any cysts, you say type = cyst. \n"
    "2- Assign type = metastasis if the %(organ)s tumor is described as a metastasis originating from a cancer in another organ. If the report mentions metastatic cancer in **another organ** along with lesions in the %(organ)s, classify the %(organ)s lesions as type = metastasis unless explicitly stated otherwise. Determine certainty based on the level of confidence expressed in the report. \n"
    "3- Try using terms from the tumor type list. E.g., if the list says 'Pancreatic Ductal Adenocarcinoma (PDAC)' and the report mentions 'adenocarcinoma in the pancreas', say type = Pancreatic Ductal Adenocarcinoma (PDAC). \n"
    "%(extra_info)s"
    "\nCertainty: "
    "Certainty of the tumor type, according to the report. If a report mentions a tumor type without demonstrating uncertainty, say certainty = certain. "
    "If the report expresses strong confidence in tumor type, say certainty = high. "
    "If the report mentions a tumor type but expresses significant uncertainty about it, say certainty = low. "
    "If the report does not mention the tumor type, say certainty = U. \n"
    "\nJustification: "
    "Besides filling the template, justify your answer, carefully mentioning sentences of the report. "
    "Explain from which sentences you got each size, location, and type.\n"
    "Provide me a syntactic analysis of the report sentences mentioning %(organ)s tumor sizes. In this analysis, explain which measurement refers to which tumor, if the measurement is current or past, and if the corresponding tumor is malignant or benign.\n"
    "I will provide an example of a report and a correct answer for a %(organ)s tumor:\n"
    "Example report: \n"
    "%(example_report)s \n"
    "Example answer: \n"
    "%(example_answer)s \n"
    "\n"
    "End of the example. \n"
    "\n"
)



pathologyReportPancreas="""
"COH202311270477
Status: Final result  
Dx: Pancreatic mass  
0 Result Notes
Component	
Final Diagnosis
A. PANCREAS, PANCREATIC NECK MASS, EUS GUIDED FINE NEEDLE BIOPSY X 4 PASSES:
- Adenocarcinoma.
Comments/Recommendations	
The clinical finding of a pancreatic mass and elevated CA 19-9 (>2000) is noted.  The morphology on the smears and cell block sections are consistent with adenocarcinoma.  
 
A cell block is prepared and examined microscopically in the evaluation of this case. The cellblock sections contain rare malignant single cells or small nests infiltrating the desmoplastic stroma (low to insufficient quantity for ancillary testing).
Rapid Evaluation	
A. PANCREAS, PANCREATIC NECK MASS, EUS GUIDED FINE NEEDLE BIOPSY:
EVALUATION BY TELECYTOLOGY, EPISODE #1:  Pass #1 - Adequate (1520) 
Gross Description	
A. PANCREAS, PANCREATIC NECK MASS, EUS GUIDED FINE NEEDLE BIOPSY X 4 PASSES:
Four passes of bloody material are taken by the clinician from which the cytology personnel prepared 6 direct smears on site.  Two formalin-fixed cell blocks are prepared from the needle rinse.
Pap X 3; DiffQuik X 3; Cell Block X 2 (4 stained slides)
Clinical Information	
pancreatic mass
"""

pathologyReportPancreasAnswer="""
### Filled Template:

Pancreas tumor 1: type = Pancreatic Ductal Adenocarcinoma (PDAC); certainty = certain; size = U; location = neck;

---

### Justification:

1. **Type**:  
   - The report explicitly states the tumor type as "adenocarcinoma" in the "Final Diagnosis." Based on the tumor type list provided, "Adenocarcinoma" in the pancreas corresponds to "Pancreatic Ductal Adenocarcinoma (PDAC)."
   - Certainty is "certain" because the report provides a definitive diagnosis supported by morphological findings on smears and cell block sections.

2. **Size**:  
   - The report does not mention the size of the tumor; therefore, size is marked as "U" (unknown).

3. **Location**:  
   - The tumor is described as being in the "pancreatic neck," which specifies the location.

4. **Additional Details**:  
   - The report also notes elevated CA 19-9 levels (>2000), which is consistent with pancreatic adenocarcinoma.
   - The comment mentions "rare malignant single cells or small nests infiltrating the desmoplastic stroma," reinforcing the diagnosis.

---

### Syntactic Analysis of Tumor Size Sentences:

- **Sentence 1**: No sentence in the report specifies the size of the pancreatic mass.  
  - As no size details are available, it is marked as "U" in the template.

- **Tumor Classification**:  
  - The tumor is malignant, as explicitly stated in the diagnosis of "adenocarcinoma" and supported by the clinical finding of elevated CA 19-9 levels and pathological observations.

Let me know if you need further clarification or additional details.
"""


instructions0ShotSizenTypeMultiOrgan = (
   "Instructions: The radiology report below possibly mentions one or more focal lesions (e.g., tumor, mass, nodule, cyst). "
   "Read it carefully, paying special attention to the findings, clinical history, and impressions sections (if available). \n"
   "Your task is to list the types, certainty of lesion type, sizes, organ, locations and attenuation of all lesions in the report. \n"
   "Fill out the template below, using one line per lesion (you may add or remove lines from the template): \n"
   "lesion 1: type = _; certainty = _; size = _; organ = _; location = _; attenuation = _; \n"
   "lesion 2: type = _; certainty = _; size = _; organ = _; location = _; attenuation = _;\n"
   "... \n"
   "If you are absolutely sure the report mentions no lesion, do not use the template. Instead, reply with: 'No lesions mentioned.' and justify why you are sure the report mentions no lesion. \n"
   "Consider the following instructions: \n"
   "\n A - What is a Lesion:\n"
   "A lesion is any focal abnormality, including masses, cysts, or areas of altered density (hyperdense, hypodense, or isodense).\n"
   "You must list both benign and malignant lesions. Include lesions that are confirmed as well as those that are only suspicious (in this case, use 'certainty = low' in the filled template).\n"
   "Common terms that indicate a lesion: metastasis, nodule, soft tissue, nodular thickening, tumor, lesion, mass, cyst, pseudo-cyst, neoplasm, cancer, index lesion, oncologic finding, adenoma, carcinoma, growth, abnormal thickening, focus, LI-RADS lesion, hyperdensity, hypodensity, and isodensity.\n"
   "Terms that are not a focal lesion: diverticulum (unless it is suspicious for malignancy); renal/gallbladder stones (e.g., cholelithiasis); hyper/hypoenhancing fluid collection not caused by a cyst or mass (e.g., cause by abscess, or postoperative changes); if the patient has cancer history but the tumor was surgically removed (e.g., whipple procedure for pancreatic tumors) and there is no current evidence of the tumor, do not list it;"
   "\n B - Size:\n"
   "1- When to use numbers:\n"
   "Reports can write the size of the lesion in 1D, 2D, or 3D measurements, and you should use the same standards used in the report. "
   "Write 1D measurements as: 15 mm; 2D measurements as: 15 x 10 mm; and 3D measurements as: 40 x 30 x 30 mm. You may use either cm or mm, but you MUST WRITE in each line of the filled template the unit you are using (cm or mm). If a report does not specify the unit, assume it is mm. \n"
   "Say size = U if the report does not specify the size of a lesion. \n"
   "IMPORTANT — organ size is NOT lesion size. Do NOT use the overall dimensions of an enlarged organ (e.g., 'prostate measures 6 x 5 x 4 cm', 'prostate enlarged to 5.5 cm', 'prostatic volume 80 cc', 'AP diameter 4.5 cm', 'splenomegaly measuring 17 cm', 'hepatomegaly with craniocaudal length 22 cm') as a lesion size. The lesion size must come from the lesion's own measurement (e.g., '1.4 cm hypodense lesion in the peripheral zone' -> size = 14 mm). If the report describes a lesion but only provides the organ's dimensions (and no focal-lesion measurement), say size = U. This is especially common for the prostate, where prostate cancer is almost always focal (a small lesion within a zone of the gland, usually the peripheral zone), NOT the whole gland — so a gland diameter like '5.2 cm in axial dimension' is not a tumor size. Another example, this time for the uterus: 'The uterus measures 8.8 cm x 3.8 cm x 5.3 cm. The endometrial cavity is distended by the presence of a soft tissue mass that measures up to approximately 2.8 cm in thickness.' -> the 8.8 x 3.8 x 5.3 cm is the uterus (organ) size and must NOT be used; the lesion is the endometrial soft tissue mass, so size = 2.8 cm. \n"
   "2- When to use `size = tiny` or `size = massive`:\n"
   "If a reports describes a lesion without numeric diameters, but describes the lesion with adjectives like tiny, small, large, or massive, you **must** include one entry with 'size = tiny' or 'size = massive' for that lesion. However, always prefer using the lesion diameter, if provided. \n"
   "If the reports does not provide the diameter nor any size-related adjective, say size = U. \n"
   "3- When to use `size = multiple`:\n"
   "If the report mentions multiple lesions in an organ **but does not give an exact count**, you **must** include one entry with 'size = multiple' for that organ.\n"
   "Only use `size = multiple` when you cannot determine the number; otherwise, create individual entries for each described lesion.\n"
   "Handling reports with unknown lesion counts AND some described lesions:\n"
   "If the report says there are multiple/innumerable/many lesions in an organ **and** then describes a few of them (e.g., gives their size), your output should include:\n"
   "One row describing each **explicitly described** lesion\n."
   "One additional row with `size = multiple` to represent the unspecified lesions.\n"
   "Example:\n"
   "Report: A 2 cm metastasis in liver sub-segment 3, and multiple other small metastases in the liver.\n"
   "Your output:\n"
   "lesion 1: type = metastasis; certainty = high; size = 2 cm; organ = liver; location = segment 3; attenuation = U;\n"
   "lesion 2: type = metastasis; certainty = high; size = multiple; organ = liver; location = U; attenuation = U;\n"
   "Key rules:"
   "1. Always include individual entries for every lesion that is specifically described in the report.\n"
   "2. Whenever the report indicates multiple unspecified lesions in an organ, also include one entry with `size = multiple` for the organ.\n"
   "\n C - Organ:\n"
   "Organ is the organ where the lesion is located. Use standard organ names, like: liver, pancreas, kidney, spleen, colon, pelvis, adrenal gland, bladder, gallbladder, breast, stomach, lung, esophagus, uterus, bone, prostate, and duodenum. \n"
   "Do not consider the 'GI-Tract' as an organ. Instead, try to localize the tumor in one of these GI-tract organs: esophagus, stomach, duodenum, small intestines or colon. You can consider 'organ = colon' for rectal lesions. You can consider 'organ = esophagus' for lesions in the esophagus-gastric junction. \n"
   "\n D - Location:\n"
   "For location, check if the report specifies the sub-segment or part of the organ where the lesion is. If it specifies: for liver, choose location as segment 1/2/.../8. For the pancreas choose the pancreas head/body/neck/tail/uncinate process. For the kidney: left kidney/right kidney. For other organs, just check if the report mentions some type of organ region or sub-segment. \n"
   "Say location = U if the report does not specify the intra-organ location of a lesion. \n"
   "If a single lesion is in more than one location, you can say both. E.g., location = liver segment 4/5. \n"
   "\n E - Type:\n"
   "If the report provides lesion type, inform it. \n"
   "Otherwise, say type = U if the report does not specify the type of the lesion. \n"
   "Follow these rules:"
   "1- If the lesion type is not specified in findings, you may deduce it from the clinical history or impressions sections. \n"
   "2- Cyst, or cystic lesion, is a common type of lesion. For any cysts, you say type = cyst. \n"
   "3- Assign type = metastasis if the lesion is described as a metastasis (or implant) originating from a cancer in another organ, unless the report explicitly states otherwise. Determine certainty based on the level of confidence expressed in the report. \n"
   "4- If the report does not mention the lesion type, but you read that the patient has cancer in the organ where the lesion is, or has a history of malignant lesions in the organ, or the lesion is growing, or it is 'suspicious for malignancy', say type = malignant. However, do not say type = malignant if a more specific lesion type is given (like PDAC and PNET in pancreas, RCC in kidney, HCC in liver,...). \n"
   "4- If the report does not mention the lesion type, but it explicitly indicates that the lesion is likely benign (or mentions it may be one of several benign lesion types), say type = benign. However, do not say type = benign if a more specific lesion type is given (like cyst or polyp). \n"
   "Prostate: enlarged prostate is a common finding, which may be benign or malignant. Use type = 'BPH' (BENIGN) only when the report explicitly uses 'BPH', 'benign prostatic hyperplasia', 'prostatic hyperplasia', 'hyperplastic prostate', or 'prostatic hypertrophy', which are assumed benign; otherwise, use type = 'enlarged prostate' (malignant or benign) when the report describes an enlarged prostate without explicitly calling it benign; Obviously, if the report describes current 'prostate cancer', 'prostatic adenocarcinoma', 'prostatic carcinoma', or similar, use type = 'prostate cancer' (malignant). \n"
   "5- Try reporting types using their most standard name, followed by their acronym. For example, if the report mentions 'adenocarcinoma in the pancreas' or 'PDAC', say type = Pancreatic Ductal Adenocarcinoma (PDAC). \n"
   "\n F - Certainty:\n"
   "Certainty of the lesion type, according to the report. If a report mentions a lesion type in the findings, history, or impressions, without demonstrating any uncertainty, say certainty = certain. "
   "If the report expresses strong confidence in lesion type, say certainty = high. "
   "If the report mentions a lesion type but expresses significant uncertainty about it, say certainty = low. "
   "If the report does not mention the lesion type, say certainty = U. \n"
   "Do NOT use certainty = certain when the report hedges the lesion type with uncertainty words such as 'likely', 'probably', 'most likely', 'presumed', 'presumably', 'possibly', 'may represent', 'may be', 'consistent with', 'in keeping with', 'compatible with', 'suggestive of', 'suspicious for', 'concerning for', 'favored to represent', 'thought to represent', 'cannot exclude', 'cannot be excluded', 'differential includes', 'statistically likely', 'unspecified'. These words signal that the radiologist is not committing to the type. In these cases, use certainty = high for confident hedges ('likely', 'probably', 'most likely', 'consistent with', 'in keeping with', 'compatible with', 'presumed') and certainty = low for weaker hedges ('possibly', 'may represent', 'may be', 'suspicious for', 'concerning for', 'cannot exclude', 'differential includes'). Use certainty = certain only when the report states the type without any hedge (e.g., 'pancreatic adenocarcinoma', 'simple renal cyst', 'biopsy-proven HCC'). \n"
   "\n G - Attenuation:\n"
   "For each lesion, inform the attenuation if the report mentions it. You should choose one of these options: hyperenhancing, hypoenchanging, isoenhancing, hererogeneously enhancing, or U (unknown). The report may use synonyms like hypoattenuating or hyperattenuating, but you must only answer me hyperenhancing, hypoenchanging, isoenhancing, hererogeneously enhancing or U (unknown). IMPORTANT: 'hypermetabolic' (or 'FDG-avid') is a PET finding describing metabolic activity, NOT CT contrast enhancement — it does NOT automatically mean hyperenhancing on CT. Base the attenuation only on the lesion's CT density/enhancement; if the report provides only PET uptake (e.g., hypermetabolic) and no CT enhancement information, say attenuation = U. \n"
   "\n H - Justification:\n"
   "Besides filling the template, justify your answer, carefully mentioning each section of the report if present: history, findings, and impressions. "
   "Explain from which sentences you got each size, location, and type.\n"
   "Some reports may refer to past measurements (using words like previously, before, or giving dates). Ignore previous measurements. "
   "Provide me a syntactic analysis of the report sentences mentioning lesion sizes. In this analysis, explain which measurement refers to which lesion, if the measurement is current or past, and if the corresponding lesion is malignant or benign.\n"
   "I will provide an example of a report and a correct answer for a %(organ)s lesion:\n"
   "Example report: \n"
   "%(example_report)s \n"
   "Example answer: \n"
   "%(example_answer)s \n"
   "\n"
   "End of the example. \n"
   "\n"
)

summary_of_terms_pancreas="""Adenocardinoma or similar terms indicate Pancreatic Ductal Adenocarcinoma (PDAC). Neuroendocrine tumor or similar terms indicate Pancreatic Neuroendocrine Tumor (PNET).
6- If the report mentions 'pancreatic cancer' or a diagnosis of 'pancreatic adenocarcinoma' and metastases in other organs (e.g., liver), classify the pancreatic lesion as the primary tumor (e.g., type = cancer or type = Pancreatic Ductal Adenocarcinoma (PDAC)). The pancreatic tumor should not be classified as metastasis in these cases, as it is the origin of the metastatic disease. Only consider type = metastasis if the report mentions a metastatic tumor from other organ, like metastatic RCC.
7- If the report says only 'pancreatic cancer', and gives no other indication of the specific type, you should classify the tumor as type = cancer."""


report_pancreas = """
HISTORY: 81-year-old with pancreatic adenocarcinoma; evaluation of treatment response._x000D_
FULL RESULT:  Prior exam:  6/3/2020_x000D_
TECHNIQUE: Volumetric imaging was obtained of the chest, abdomen and pelvis on a MDCT scanner and reconstructed at 2.5 and 5 mm slice thickness. The images were acquired precontrast through the abdomen and post bolus IV infusion of 125 mL of nonionic contrast (Isovue 370). The postcontrast scans of the abdomen were obtained during the arterial and portal venous phases. The patient received oral contrast. In addition, coronal and sagittal reconstructions of the postcontrast images as well as axial MIP images of the chest were performed.  Up-to-date CT equipment and radiation dose reduction techniques were employed. CTDIvol: 8.1 - 16.8 mGy. DLP: 1174 mGy-cm._x000D_
FINDINGS:_x000D_
LOW NECK: The low neck structures remain unremarkable. No base of neck adenopathy is noted._x000D_
CHEST: _x000D_
LUNGS AND PLEURA:  There are stable scattered areas of subsegmental atelectasis in both posterior lung bases. There is slight increase in size of a now 4 mm nodule in the left lung base (12/245) versus less than 3 mm on the prior exam (12/237). No other significant pulmonary lesions, infiltrates, or pleural effusion is noted._x000D_
OTHER THORACIC STRUCTURES:  No axillary adenopathy or chest wall lesions identified. There is a stable central venous Port-A-Cath via the right anterior chest wall with associated catheter terminating at or near the cavoatrial junction._x000D_
_x000D_
ABDOMEN:_x000D_
ABDOMINAL ORGANS (INCLUDING BILIARY TREE): The liver remains within normal limits in size and appearance. _x000D_
Spleen is unremarkable and unchanged._x000D_
Previous cholecystectomy. There is a stable common bile duct stent with associated persistent mild pneumobilia._x000D_
The mass in the head/uncinate process of the pancreas is apparently mildly increased in size at 26 x 25 mm (12/94) versus 24 x 21 mm on the prior exam (8/176). The mass continues to abut the medial margin of the common bile duct as well as the superior mesenteric vein. There is persistent involvement of the celiac and superior mesenteric axes. There is also persistent significant pancreatic duct dilation with secondary atrophy of the pancreatic body and tail. No new pancreatic lesion is noted._x000D_
No adrenal lesion is identified._x000D_
No suspicious renal lesion or hydronephrosis demonstrated. There are a few small stable bilateral renal cysts._x000D_
Stomach and bowel loops remain within normal limits in appearance allowing for the inclusion of large bowel loops within a large ventral abdominal wall hernia without evidence of incarceration or obstruction._x000D_
LYMPH NODES, MESENTERY, AND OMENTUM: No retrocrural, retroperitoneal, mesenteric, iliac, pelvic, or inguinal adenopathy is noted. No omental or serosal lesion is identified._x000D_
_x000D_
PELVIS: Bladder is grossly intact and unchanged. Previous prostatectomy. No pelvic mass identified._x000D_
OTHER ABDOMINAL AND PELVIC STRUCTURES: No ascites or fluid collection noted. No venous thrombosis is seen. There is also evidence of previous anterior abdominal wall surgery consistent with hernia repair._x000D_
BONES: No suspicious bone lesion identified. There is a stable left hip arthroplasty which is grossly intact._x000D_
_x000D_
IMPRESSION:_x000D_
1.	 Probable mild increase in size of the ill-defined presumed tumor mass in the head/uncinate process of the pancreas. There is associated presumed obstruction of the pancreatic duct associated with atrophy of the pancreatic body and tail._x000D_
2.	Stable common bile duct stent in situ with associated pneumobilia._x000D_
3.	Previous cholecystectomy._x000D_
4.	Stable large ventral abdominal wall hernia containing a portion of the transverse colon. There has also been previous anterior abdominal wall surgery presumably representing hernia repair._x000D_
5.	Previous prostatectomy._x000D_
6.	Stable left hip arthroplasty.
"""

answer_pancreas = """
Pancreas tumor 1: type = Pancreatic Ductal Adenocarcinoma (PDAC); certainty = certain; size = 26 x 25 mm; location = head;

Justification:
- Type: The tumor type is explicitly mentioned as "pancreatic carcinoma" in the history section ("81-year-old with pancreatic carcinoma; evaluation of treatment response").
- Certainty: The report history section demonstrates certainty in the tumor type by directly referring to it as "pancreatic carcinoma" without any language suggesting uncertainty, and mentioning treatment for the cancer.
- Size: The current size is explicitly mentioned in the findings as "26 x 25 mm (12/94)" and is noted as slightly increased compared to the prior exam.
- Location: The location is described as "head/uncinate process of the pancreas" in the findings section. To choose between head, body and tail, we consider the ucinante process and neck as parts of the head.
- Past Measurements Analysis: The previous size ("24 x 21 mm") was ignored as instructed, focusing only on the current measurement. The tumor remains ill-defined but is presumed stable with persistent involvement of key pancreatic and vascular structures.
"""

answer_pancreas_multi_organ="""
Answer (template filled):

tumor 1: type = Pancreatic Ductal Adenocarcinoma (PDAC); certainty = certain; size = 26 x 25 mm; organ = pancreas; location = head/uncinate process; attenuation = U;
tumor 2: type = nodule; certainty = certain; size = 4 mm; organ = lung; location = left lung base; attenuation = U;
tumor 3: type = cyst; certainty = certain; size = tiny; organ = kidney; location = bilateral; attenuation = U;

Justification

History
- The report's history explicitly states: “81-year-old with pancreatic adenocarcinoma; evaluation of treatment response,” confirming a known pancreatic malignancy.

Findings
- Pancreas: A dominant mass in the head/uncinate process measures 26 x 25 mm (previously 24 x 21 mm). No attenuation descriptor is given, so attenuation = U.
- Lungs: “There is slight increase in size of a now 4 mm nodule in the left lung base… versus <3 mm on the prior exam.” This single, clearly identified nodule justifies tumor 2.
- Kidneys: “There are a few small stable bilateral renal cysts.” The adjective “small” indicates very small lesions without numeric measurement; we code one representative entry with size = tiny and location = bilateral for these cysts (tumor 3).
- Lymph-node measurements are described as “borderline/enlarged” or “stable” nodes, but they are not designated as metastatic tumors, so they are excluded from the tumor list.

Impression
- Impression 1 repeats the “probable mild increase” of the pancreatic head/uncinate tumor, supporting its malignant nature.
- No remarks contradict the inclusion of the lung nodule or renal cysts; both remain stable/incidental findings appropriate for listing as focal lesions.

Certainty Assignments
- For tumor 1, “pancreatic adenocarcinoma” is stated unequivocally → certainty = certain.
- For tumor 2, the term “nodule” is definitive and lacks uncertainty language → certainty = certain.
- For tumor 3, the lesions are explicitly called “cysts,” a definitive benign entity → certainty = certain.

Syntactic Analysis of Tumor-Size Sentences
- “The mass in the head/uncinate process of the pancreas is … 26 x 25 mm…”  
  - Current measurement 26 x 25 mm → tumor 1 (malignant PDAC).  
  - Earlier size 24 x 21 mm is historical and ignored for template size.
- “There is slight increase in size of a now 4 mm nodule in the left lung base…”  
  - Current measurement 4 mm → tumor 2 (lung nodule).  
  - The phrase “<3 mm on the prior exam” is historical and ignored for size entry.
- “There are a few small stable bilateral renal cysts.”  
  - No numeric size; adjective “small” → coded as size = tiny → tumor 3 (renal cysts).  
  - Multiple cysts but unspecified count → represented by a single line with location = bilateral.

No additional focal lesions are described; therefore, only the three tumor lines above are required.
"""

report_liver="""
CT CHEST PELVIS W CONTRAST ABDOMEN W WO CONTRAST
ORDERING HISTORY:  research. Right scapular mass. Hepatocellular carcinoma.
COMPARISON: MRI chest with contrast 3/14/2024. CT chest with contrast 3/8/2024. MRI abdomen pelvis 2/27/2024
TECHNIQUE: Multidetector CT of the abdomen without IV contrast followed by CT chest, abdomen and pelvis from the lung apices through the ischial tuberosities. 125 ml of intravenous contrast was administered during this examination.  Multiphasic imaging 
was obtained.  Oral contrast was administered during this examination. 3D imaging was reviewed at the PACS workstation.
Up-to-date CT equipment and radiation dose reduction techniques were employed. CTDIvol: .8 - 4.4 mGy. DLP: 136 mGy-cm.
Findings:
CHEST:
Chest Wall: There is a briskly enhancing, heterogeneous mass with associated osseous destruction arising from the medial right scapula. Enhancing component, measures 6.8 x 3.6 cm (TR, AP) measured on (11/34), previously measuring 7.0 x 4.7 cm. Currently 
this measures approximately 6.4 cm craniocaudad.
Thoracic inlet \T\ Thyroid: No significant abnormalities.
Mediastinum/hilum: The hila are normal in appearance.  No mediastinal masses or adenopathy is seen.  The visualized tracheobronchial tree is unremarkable.  The visualized esophagus is normal.
Heart: The heart is normal is size. There is no pericardial effusion.  The cardiac vessels demonstrate minimal or no calcification.
Aorta and Great Vessels:No aneurysm or dissection of the aortic arch or thoracic aorta. The proximal great vessels demonstrate no significant stenoses.
Lungs: There is a 3.9 x 2.9 cm heterogeneous mildly enhancing pleural-based mass along the posterior right lower lobe. On MRI of the chest dated 3/14/2024, this measured 3.1 x 2.4 cm.
Pleura: There are no pleural effusions, pneumothorax, or hemothorax.
ABDOMEN:
Liver: Status post right hepatic lobectomy with hypertrophied left lobe. Heterogeneous, mildly arterial phase enhancing lesion in segment 2, with washout on the venous phase, measuring 1.4 x 1.4 cm (11/174) highly concerning for HCC, LI-RADS 4 lesion.
Gallbladder and Biliary Tree: Gallbladder is surgically absent. No intrahepatic or extrahepatic biliary dilation.
Spleen: Enlarged measuring 17.5 cm craniocaudad. No focal abnormality.
Pancreas: No focal pancreatic lesion or ductal dilatation.
Adrenal Glands: Within normal limits. 
Kidneys, Ureters \T\ Collecting System: Symmetric enhancement. No hydronephrosis. No suspicious renal lesion. No calculi. 
Bladder: Unremarkable given degree of distension.
Peritoneum, Bowel and Mesentery: The stomach is grossly normal in appearance. Small bowel and colon are normal in caliber and distribution.  The appendix is normal caliber without findings of acute appendicitis identified. No pneumatosis. No frank 
peritoneal carcinomatosis.
Ascites: Trace free fluid in the pelvis. Trace fluid in the perihepatic region.
Abdominal and Pelvic Lymphadenopathy: No pathologic lymphadenopathy.
Abdominal Wall: No acute or suspicious finding.
Vasculature: The visualized abdominal aorta is normal in caliber. Abdominal and pelvic vessels demonstrate normal enhancement. No advanced atherosclerotic disease or aneurysm. Gastroesophageal junction varices.
Pelvic Organs: No significant abnormality is visualized.
Musculoskeletal: Destructive heterogeneously enhancing right medial scapular mass, consistent with metastatic disease. No additional lesion is readily identified.

IMPRESSION:
1.  Postsurgical changes of right hepatic lobectomy. Arterial phase enhancing lesion with washout in the left hepatic lobe, segment 2, highly concerning for hepatocellular carcinoma, not visualized on prior MRI of the abdomen and pelvis.
2.  Destructive lesion in the medial right scapula with large enhancing soft tissue component, marginally decreased in size when compared to prior study.
3.  Enlarging pleural-based right lower lobe mass, consistent with metastasis.
4.  Features of portal hypertension including splenomegaly and gastrohepatic esophageal varices."""

answer_liver= """
Liver tumor 1: type = Hepatocellular Carcinoma (HCC); certainty = high; size = 1.4 x 1.4 cm; location = segment 2;

Justification:
- Type: The type is identified as "hepatocellular carcinoma (HCC)" in the impression section ("Arterial phase enhancing lesion with washout in the left hepatic lobe, segment 2, highly concerning for hepatocellular carcinoma") and findings section ("Heterogeneous, mildly arterial phase enhancing lesion in segment 2, with washout on the venous phase... highly concerning for HCC, LI-RADS 4 lesion").
- Certainty: The certainty is "high" as the report consistently uses strong terms such as "highly concerning for HCC" and provides a LI-RADS 4 classification, which strongly indicates malignancy.
- Size: The size is explicitly mentioned as "1.4 x 1.4 cm (11/174)" in the findings section.
- Location: The location is "segment 2," clearly specified in both the findings and impression sections.
- Past Measurements Analysis: There is no mention of a previous measurement for this lesion. The lesion was not visualized on the prior MRI, confirming its new appearance. This supports its relevance in the current context.
"""
report_kidney="""HISTORY: Non-small cell lung cancer.
COMPARISON: None available   
TECHNIQUE: A volumetric CT of the chest, abdomen, and pelvis was obtained with contrast. Precontrast images were acquired through the liver. 125 cc of Isovue-370 was administered intravenously without complications. Oral contrast was also administered. 
Multiplanar reconstructions and MIP images were submitted for review. 
Up-to-date CT equipment and radiation dose reduction techniques were employed. CTDIvol: 7.4 - 7.7 mGy. DLP: 880 mGy-cm.
 
FULL RESULT: 
 
FINDINGS:
 
CHEST:  
The heart is normal in size.  No pericardial effusion or thickening is identified.
The thoracic aorta is normal in caliber. The superior vena cava, just above the cavoatrial junction is narrowed. The left brachiocephalic vein is thrombosed. Collateral vessels in the back and left chest wall.
The thyroid gland is small and contains a 4 mm hypodense nodule in the left lobe (6-12). The trachea is unremarkable. There are esophageal mucosal varices.
There is no mediastinal, hilar or axillary lymphadenopathy.
There is a masslike consolidation in the right lower lobe measuring approximately 7.1 x 5.1 cm (6-89). There is a larger area of surrounding groundglass opacity and interlobular septal thickening, which measures up to 12.5 cm in maximal axial dimension, 
is concerning for lepidic growth of tumor.
There are scattered other pulmonary nodules, which are nonspecific.
No pleural effusion or pneumothorax is identified.
There is a 6 mm nonspecific subcutaneous nodule in the anterior left chest wall (6-26) at the level of the first costosternal joints.
 
ABDOMEN/PELVIS:  
The liver is nodular in contour. No focal liver lesion is identified.
The portal vein is patent.
There is no intra or extrahepatic biliary dilatation.
The gallbladder is contracted and contains multiple calcified gallstones.
The spleen has been resected. 
The pancreas enhances homogeneously. The main pancreatic duct is not dilated.
The adrenal glands are normal in thickness.
The kidneys are lobulated in contour. There is a 5.2 cm cyst in the lower pole of the left kidney (6-184) with punctate mural calcifications. The subcentimeter hypodensities in the right kidney have not significantly changed in size. There are punctate 
calculi in the right kidney. No evidence of hydronephrosis or hydroureter is identified. 
The bowel is normal in caliber and wall thickness. There are a few scattered colonic diverticula. The appendix is not identified; no secondary signs of acute appendicitis are noted.
The bladder wall appears mildly thickened. The median lobe of the prostate gland indents the base of bladder.
There is no evidence of abdominal or pelvic lymphadenopathy.
The dense calcification in the left upper quadrant (6-137) and peripherally calcified nodule in the left lower quadrant, adjacent to the descending colon (6-226),
The abdominal aorta is normal in caliber. Calcified and noncalcified atherosclerotic plaque is seen in the abdominal aorta and bilateral iliofemoral arteries. The superficial femoral veins appear occluded.
No evidence of free fluid is identified.
 
MUSCULOSKELETAL: 
No suspicious osseous lesions is identified, although evaluation is limited by demineralization.
There is moderate to severe disc degeneration in the lumbar spine with multilevel vacuum disc phenomenon.
 
IMPRESSION:
Large mixed solid and groundglass lesion in the right lower lobe has appearance of adenocarcinoma with lipidic tumor growth.
Scattered nonspecific pulmonary nodules. Attention on follow-up.
Postoperative changes related to a splenectomy.
Significant atherosclerosis of the superficial femoral veins with suggestion of occlusion and/or high-grade narrowing.
Lesion of the left brachiocephalic vein and narrowing of the lower portion of the superior vena cava. Multiple collateral vessels in the chest."""

answer_kidney="""
Kidney tumor 1: type = cyst with mural calcifications; certainty = certain; size = 5.2 cm; location = lower pole of the left kidney;

Justification:
- Type: The type is identified as "cyst with mural calcifications" based on the findings section, which describes "a 5.2 cm cyst in the lower pole of the left kidney with punctate mural calcifications." This terminology suggests a benign lesion.
- Certainty: The certainty is "certain" as the report describes the lesion explicitly as a cyst and provides no indication of malignancy or uncertainty.
- Size: The size is explicitly mentioned as "5.2 cm" in the findings section.
- Location: The location is clearly described as "lower pole of the left kidney" in the findings section.
- Past Measurements Analysis: No previous measurements for this cyst are provided, so there is no need to compare with prior data.
"""

liver_locationsliver_segments = "Segment I (Caudate Lobe), Segment II, Segment III, Segment IV, Segment V, Segment VI, Segment VII, Segment VIII"
pancreas_subdivisions = "Head, Uncinate Process, Neck, Body, Tail"
kidney_subdivisions = "Right Kidney Renal Cortex, Right Kidney Renal Medulla, Right Kidney Renal Pyramids, Right Kidney Renal Papilla, Right Kidney Renal Columns, Right Kidney Renal Pelvis, Right Kidney Minor Calyces, Right Kidney Major Calyces, Right Kidney Hilum, Left Kidney Renal Cortex, Left Kidney Renal Medulla, Left Kidney Renal Pyramids, Left Kidney Renal Papilla, Left Kidney Renal Columns, Left Kidney Renal Pelvis, Left Kidney Minor Calyces, Left Kidney Major Calyces, Left Kidney Hilum"
organ_part={"liver":liver_locationsliver_segments,"pancreas":pancreas_subdivisions,"kidney":kidney_subdivisions}

# Liver Tumors
benign_liver = "Hepatic Hemangioma (HH), Focal Nodular Hyperplasia (FNH), Bile Duct Adenoma, Simple Liver Cyst (SLC), Cyst"
malignant_liver = "Hepatocellular Carcinoma (HCC), Cholangiocarcinoma (CCA), metastasis"
both_liver = "Hepatic Adenoma (HA), Mucinous Cystic Neoplasm (MCN)"

# Pancreas Tumors
benign_pancreas = "Serous Cystadenoma (SCA), Cyst"
malignant_pancreas = "Pancreatic Ductal Adenocarcinoma (PDAC), Mucinous Cystadenocarcinoma (MCC), metastasis"
both_pancreas = "Mucinous Cystadenoma (MCA), Intraductal Papillary Mucinous Neoplasm (IPMN), Solid Pseudopapillary Neoplasm (SPN), Pancreatic Neuroendocrine Tumor (PNET)"

# Kidney Tumors
benign_kidney = "Renal Oncocytoma (RO), Angiomyolipoma (AML), Simple Renal Cyst, Cyst"
malignant_kidney = "Renal Cell Carcinoma (RCC), Transitional Cell Carcinoma (TCC), Wilms Tumor (Nephroblastoma), metastasis"
both_kidney = "Cystic Nephroma (CN), Multilocular Cystic Renal Neoplasm of Low Malignant Potential (MCRNLMP)"

observationsV0=(" Follow these rules for interpreting radiology reports, and always check the entire report carefully, including any clinical history (especially of cancer), the findings section (if present) and the impressions section (if present): \n "
              "1- 'unremarkable' means that an organ has no tumor. \n "
              "2- Multiple words can be used to describe benign and malignant tumors, such as metastasis, tumor, lesion, mass, cyst, neoplasm, growth and cancer. Consider any lesion, hyperdensity or hypodensity a tumor, unless the report explicitly says that it is something else. Examples of lesions that are not tumors: ulcers, wounds, infections, inflammations or scars.\n"
              "3- You should consider that a certain tumor is certainly benign if this information is explicit in the report (ususally, in the findings or impressions). Examples of benign tumors are: cysts (SLC, PLD, etc.), hepatic hemangioma (HH), Focal Nodular Hyperplasia (FNH), Bile Duct Adenoma, Serous Cystadenoma (SCA), Mucinous Cystadenoma (MCA), Intraductal Papillary Mucinous Neoplasm (IPMN), Solid Pseudopapillary Neoplasm (SPN), . \n "
              "4- You should consider that a certain tumor is certainly malignant if this information is explicit in the report. Examples of benign tumors: . \n "#oncologic findings, index lesions
              #we may have both malignant and benign
              #I want the largest malignant tumor
              #If it is a cancer patient
              "4- Organs never mentioned in the report have no tumors. \n "
                "5- Treat anything that is 'too small to characterize' or uncertain as 0. \n "
                "6- Renal calculi, nephrolithiasis and renal stones are not tumors. \n "
                "7- Organs with no tumor but other pathologies should be reported as 0.")

observations=("")

#, hypodensity and hyperdensity, too small to characterize: uncertain

instructions1ShotFast=("Instructions: Discover if a CT scan radiology report indicates the presence "
                   "of liver tumors, pancreas tumors or kidney tumors. Output binary labels for "
                   "each of these categories, where 1 indicates tumor presence and 0 tumor absence. "
                   "Example: liwhatsappver tumor=1; kidney tumor=0; pancreas tumor=0. "
                   "Answer with only the labels, do not repeat this prompt. "
                   "Report 1 is an example for you, and its labels are provided. "
                   "I want you to give me the labels for Report 2.\n ")

instructionsFewShotFast=("Instructions: Discover if a CT scan radiology report indicates the presence "
                   "of liver tumors, pancreas tumors or kidney tumors. Output binary labels for "
                   "each of these categories, where 1 indicates tumor presence and 0 tumor absence."
                   "Example: liver tumor=1; kidney tumor=0; pancreas tumor=0. "
                   "Answer with only the labels, do not repeat this prompt. "
                   "I will provide a %(examples)s reports and labels as examples for you. "
                   "I want you to give me the labels for Report %(last)s.\n ")

question="Binary labels for Report%(last)s:"



time_machine_solver = ("I am sending two CT radiology reports with respective dates. The first report is from an earlier exam, expressing either no lesion or uncertainty about the presence or nature (benign or malignant) of a %(organ)s lesion (or lesions). "
                        "The second report is from a more recent exam, and it indicates more clearly the presence of a malignant tumor in the %(organ)s. Read both reports carefully, paying attention to the findings, impressions, and clinical history sections (if present). "
                        "Your task is to determine if any %(organ)s lesion reported in the first report is very likely the same as a malignant tumor in the second report. "
                        "Report 1 (earlier exam, %(date1)s): \n"
                        "%(report1)s \n"
                        "Report 2 (more recent exam, %(date2)s): \n"
                        "%(report2)s \n"
                        "Fill out the template below by answering yes, no (if the lesion in report 1 is clearly not the same as in report 2), or uncertain: \n"
                        "very likely malignancy in %(organ)s in the first exam = _ \n"
                        "In case you answer yes, also provide the location and size of the likely malignant lesion (or lesions) in the first report. Do so by filling out the template below, using one line per malignant tumor in the %(organ)s (you may add or remove lines from the template). Substitute the first _ in each line by the the tumor size, and the second by its location (Use 'U' if the report doesn’t specify size or location): \n "
                        "%(organ)s malignant tumor size = _; location = _;\n"
                        "%(organ)s malignant tumor size = _; location = _;\n"
                        "... \n"
                        "I am only interested in measurements that are 'current' at the time of report 1. Reports can write the size of the tumor in 1D, 2D or 3D measurements, and you should use the same standards used in the report."
                        "Write 1D measurements as: 15 mm; 2D measurements as: 15 x 10 mm; and 3D measurements as: 40 x 30 x 30 mm. You may use either cm or mm, but you MUST WRITE in each line of the filled template the unit you are using (cm or mm). If a report does not specify the unit, assume it is mm. \n"
                        "For location, chose one of these options for each tumor: %(organ_locations)s \n"
                        "In your justification for malignancy, explain why a lesion in report 1 is likely the same as in report 2. Refer to relevant sections (findings, clinical history, impressions), and carefully check tumor location. \n"
                        "If you provide measurements, also provide a synthatic analysis of the report sentences mentioning %(organ)s tumor sizes. In this analysis, explain which measurement refers to which tumor, if the measurement is current at the time of exam 1, and if the tumor is malignant or benign.\n"
                        "Follow these rules for interpreting radiology reports: \n"
                        "1- If report 1 mentions absolutely no abnormality in the %(organ)s, answer 'very likely malignancy in %(organ)s in the first exam = no' \n"
                        "2- Some words are only used for describing malignant tumors, for example: metastasis, cancer, growing, or any oncologic lesion and index lesion in cancer patients. \n"
                        "Reports may sometimes mention the specific tumor type. In the %(organ)s, benign tumors are: %(benign_tumors)s. \n "
                        "Malignant %(organ)s tumors are: %(malignant_tumors)s. \n "
                        "Tumors that may be both benign or malignant are: %(both_tumors)s. \n "
                        "3- If the report does not mention that the tumor is benign or does not specify lesion type, but the tumor is growing in relation to a past measurement, consider it malignant. \n"
                        "4- Especially check if the location of a malignant tumor in report 2 matches a lesion in report 1. \n"
)

compare_reports="""
You are an expert radiologist. Your task is to compare two radiology reports—Report A and Report B—and decide whether they are the same report (possibly annonymized in a different way or with parts of the text removed) or two different exams (referring to different patients/exams).

Rules:

1. Reports are the same if:  
   - Apart from redacted or removed sections and randomized dates, **all remaining sentences and phrases match exactly**, down to the same wording and order.  
   - There is **no paraphrasing**, the same words are used to describe the same findings in both reports.

2. Reports that are actually the same report may have a few differences, which you must ignore and still consider them the same:
   - Ignore differences in patient identifiers, dates, metadata, or anonymized tags. For example, annonymization may randomly change dates, and dates may not match across reports. It may also remove words.
   - Ignore gaps where text was removed (e.g. “[**]” or similar placeholders).
   - One of the repors may be missing sections like IMPRESSION or CLINICAL HISTORY, this is not a problem, they can still be the same report.

Considering these rules, are the two reports below actually the same report? Carefully read both reports and compare them, considering the rules above.

Carefully justify your answer, citing parts of the reports that do not match in case of a mismatch. After your justification, include in yout answer the following template, filled by substituting _ by yes or no:

same report = _;

Here are the reports:

Report 1:

%(report1)s \n

Report 2:

%(report2)s \n
"""

get_tumor_slices_old="""
You are an expert radiologist. Your task is to carefully analyze a radiology report and determine the image and series numbers for tumors describes in the report.
The report may describe one or multiple tumors, and the image and series numbers may appear for all, none or some of the tumors.
To make your task easier, I have already identified the tumors in the report, and I will provide you with a list of tumor descriptions. I used tumor IDs to identify each tumor, and each description includes the tumor id, and possibly the tumor location, size and attenuation. 
The tumor ID is not present in the report. Therefore, you should use the tumor description to match the tumors in my lists to the tumors in the report. Then, for each tumor, you should find the image and series numbers in the report, if they are present.
Match each tumour description to sentences in FINDINGS using all available information in my list (organ > location  > size > attenuation). Ignore case and punctuation. Remember that my list only provides measurements in mm, but the report may present measurements in cm.  
Reports may describe series number and image number in different ways. Usually, descriptions come right after the tumor is mentioned. Common ways to describe series and image number are exemplified below:
- Reports may write Se/Im to indicate series/image: "Se/Im 303/55 : Segment 6 : Hypodense : 3.6 x 3.3 cm : Previously 4.0 x 3.0 cm"; "AI1:  Segment 6 : 1.2 cm (Se/Im 6/73), previously 0.6 cm"
- Reports may explicitly write series and image: "AI1: Hepatic segment 7 : Now measuring 2.7 x 2.5 cm (series 303, image 32)"
- Reports may not explictly say which number is series and which is image, and just use a / or a : to separate the two numbers. In this case, consider the first number as series and the second as image: "Pancreas:  Grossly similar 1.4 cm pancreatic neck mass (3/63) with upstream pancreatic atrophy and mild ductal dilatation." ---> this means series 3, image 63; "Stable appearance of 1.8 x 1.7 cm heterogeneous lesion in the right lower quadrant (3:121)" ---> this means series 3, image 121.
- Reports may provide the image number only: Scattered foci of peripancreatic nodularity measuring up to 1.4 cm (image 68)"

IMPORTANT RULES:
1. NEVER guess. If either series or image number is absent or ambiguous, consider it unknown (u).  
2. Be very carefull when matching tumors in the report to the tumors in my list. If you are not sure, consider series and image numbers unknown (u) in your answer. Do not accept partial matches. Be absolutely sure that organ and sizes match.

Here is the report you must analyze. I am providing you with the FINDINGS section only. Consider I have already identified the tumors in the report, using the full report. FINDINGS:
%(findings_section)s

Below is the list of tumors I have identified in this report. I use tumor sizes for the current measurements of tumor in report. I ignored past measurements, and you should too.
In your answer, include this exact list, but fill in the image and series numbers for each tumor. 
To do so, substitute 'se' by the series number and 'im' by the image number. If some tumor does not have image or series numbers, you can substitute im or se by u (meaning unknown).
Template:
%(template_to_fill)s

After filling the template, provide a detailed justification for your answer, explaining how you found the series and image numbers for each tumor, and copying from the report the sentences where you found the series and image numbers. If you could not find the series or image numbers for some tumors, explain why you could not find them.
"""


get_tumor_slices = """
You are an expert radiologist. Your task is to carefully analyze a radiology report and determine the image and series numbers for tumors described in the report.

The report may describe one o multiple tumors, and the image and series numbers may appear for all, none, or only some of the tumors.

To make your task easier, I will give you:

1. **FINDINGS** - the body of the report.  
2. **TUMOR LIST** - my pre-extracted tumor descriptions.  
   • This list *is* your template: each line already contains  
     `series = se` and `image = im`.  
   • Replace se and im with the numbers you find in the report.  
   • If a number is missing or ambiguous, write u (unknown).  
   • Never guess the series or image numbers.

To fill in the TUMOR LIST, you must match each tumor in the TUMOR LIST to tumors described in FINDINGS. Use all all available cues I provide in the list, which may include the tumor organ, location (inside the organ), size (in mm), and attenuation. Sometimes I may not be able to get some of these cues. In these cases, I will mark it as 'u' (unknown).  
Consider that my list measures tumors in millimetres (mm). Reports use mm if they do not mention the unit. But they use centimeters (in these cases, they usually write cm). 10 mm = 1 cm.

### Examples of how series / image numbers usually appear in reports"
* `(Se/Im 303/55)` series = 303, image = 55  
* `(series 303, image 32)`  
* `(3/63)` or `(3:121)` -> series = first number, image = second  
* `(image 68)` only -> series = u, image = 68  

---

## Worked examples

### Example 1  

**REPORT (reduced to only tumor-relevant sentences)**  
Liver: Numerous, at least 15, hypoattenuating hepatic lesions nearly all of which are decreased in size since 12/06. For example, lesion in the dome of segment 7 measures 2.7 x 3.0 cm (Se/Im 4/47), previously 4.8 x 4.5 cm. Lesion in segment 4A measures 4.4 x 3.4 cm (Se/Im 4/70), previously 5.2 x 4.6 cm.  
Spleen: Hypoenhancing lesion in the anterior spleen measures 1.6 x 1.0 cm (Se/Im 4/39) previously 2.0 x 1.1 cm.  
Colon: The primary colon cancer, represented by mural thickening of the distal sigmoid colon, is less conspicuous than 12/06.

**TUMOR LIST TO FILL (input / template)**  
tumor id = 1; organ = liver; location = segment 7; size = 27 x 30 mm; attenuation = hypoattenuating; series = se; image = im;  
tumor id = 2; organ = liver; location = segment 4; size = 44 x 34 mm; attenuation = hypoattenuating; series = se; image = im;  
tumor id = 3; organ = spleen; location = u; size = 16 x 10 mm; attenuation = hypoattenuating; series = se; image = im;  
tumor id = 4; organ = colon; location = u; size = u; attenuation = u; series = se; image = im;

**FILLED LIST (expected model output)**   
tumor id = 1; organ = liver; location = segment 7; size = 27 x 30 mm; attenuation = hypoattenuating; series = 4; image = 47;  
tumor id = 2; organ = liver; location = segment 4; size = 44 x 34 mm; attenuation = hypoattenuating; series = 4; image = 70;  
tumor id = 3; organ = spleen; location = u; size = 16 x 10 mm; attenuation = hypoattenuating; series = 4; image = 39;  
tumor id = 4; organ = colon; location = u; size = u; attenuation = u; series = u; image = u;

**JUSTIFICATION**
tumor id = 1; series = 4, image = 47 — extracted from: "lesion in the dome of segment 7 measures 2.7 x 3.0 cm (Se/Im 4/47)"; matched by organ (liver), location (segment 7), and size ≈ 27 x 30 mm.
tumor id = 2; series = 4, image = 70 — extracted from: "lesion in segment 4A measures 4.4 x 3.4 cm (Se/Im 4/70)"; matched by organ (liver), location (segment 4), and size ≈ 44 x 34 mm.
tumor id = 3; series = 4, image = 39 — extracted from: "hypoenhancing lesion in the anterior spleen measures 1.6 x 1.0 cm (Se/Im 4/39)"; matched by organ (spleen) and size ≈ 16 x 10 mm.
tumor id = 4; series = u, image = u — no series / image numbers are given for the sigmoid-colon tumour; therefore both values are unknown.

---

### Example 2  

**REPORT (reduced to only tumor-relevant sentences)**  
Spleen: Slightly decreased size of a hypodense lesion in the spleen measuring up to 3.1 cm (series 5, image 30) previously 2.7 cm. Unchanged small hypervascular nodule at the tip of the spleen (series 601, image 43).  
Adrenal Glands: Stable 1.5 cm left adrenal nodule (series 5, image 40).  
Kidneys: Unchanged 1.3 cm enhancing lesion in the inferior pole of the right kidney (series 5, image 79). Additional bilateral sub-centimeter renal hypodensities that are too small to characterize, stable from prior.  
Peritoneum: Mild omental fat stranding along the left anterior abdomen (series 5, image 100), new from prior. Similar sequela of prior epiploic appendagitis anterior to the descending colon (series 5, image 92). Similar small capsular nodule adjacent to the inferior spleen (series 5, image 55).
Reproductive organs: Slightly increased size of a heterogeneously enhancing mass in the uterus, now measuring 6.5 x 5.1 cm (series 5, image 133), previously 6.1 x 4.8 cm.  
Liver: Unchanged sub-centimeter hypodensities that are too small to characterize.

**TUMOUR LIST TO FILL (input / template)**  
tumour id = 1; organ = spleen; location = u; size = 31 mm; attenuation = hypoattenuating; series = se; image = im;  
tumour id = 2; organ = spleen; location = u; size = u; attenuation = hyperattenuating; series = se; image = im;  
tumour id = 3; organ = adrenal gland; location = left; size = 15 mm; attenuation = u; series = se; image = im;  
tumour id = 4; organ = kidney; location = right; size = 13 mm; attenuation = hyperattenuating; series = se; image = im;  
tumour id = 5; organ = uterus; location = u; size = 65 x 51 mm; attenuation = u; series = se; image = im;  

**FILLED LIST (expected model output)**  
tumour id = 1; organ = spleen; location = u; size = 31 mm; attenuation = hypoattenuating; series = 5; image = 30;  
tumour id = 2; organ = spleen; location = u; size = u; attenuation = hyperattenuating; series = 601; image = 43;  
tumour id = 3; organ = adrenal gland; location = left; size = 15 mm; attenuation = u; series = 5; image = 40;  
tumour id = 4; organ = kidney; location = right; size = 13 mm; attenuation = hyperattenuating; series = 5; image = 79; 
tumour id = 5; organ = uterus; location = u; size = 65 x 51 mm; attenuation = ug; series = 5; image = 133;  

**JUSTIFICATION**
tumor id = 1; series = 5, image = 30 — extracted from: "hypodense lesion in the spleen measuring up to 3.1 cm (series 5, image 30)"; matched by organ (spleen) and size ≈ 31 mm.
tumor id = 2; series = 601, image = 43 — extracted from: "hypervascular nodule at the tip of the spleen (series 601, image 43)"; matched by organ (spleen) and hyperattenuating (hypervascular) attenuation.
tumor id = 3; series = 5, image = 40 — extracted from: "stable 1.5 cm left adrenal nodule (series 5, image 40)"; matched by organ (adrenal), location (left), and size ≈ 15 mm.
tumor id = 4; series = 5, image = 79 — extracted from: "1.3 cm enhancing lesion in the inferior pole of the right kidney (series 5, image 79)"; matched by organ (kidney), location (right), enhancing nature, and size ≈ 13 mm.
tumor id = 5; series = 5, image = 133 — extracted from: "heterogeneously enhancing mass in the uterus … 6.5 x 5.1 cm (series 5, image 133)"; matched by organ (uterus) and size ≈ 65 x 51 mm.



---

### Example 3 

**REPORT (reduced to only tumor-relevant sentences)**  
Liver:  Diffusely hypoattenuating. Unchanged subcentimeter segment 2 hyperdense focus (3/32), likely a benign hemangioma.
Gallbladder: Cholelithiasis without evidence of acute cholecystitis.
Spleen:  Similar-appearing hypoattenuating lesions, some of which are too small to characterize, which have been gradually increasing in size since at least 9/13/2021. The largest measures up to 1.8 cm.
Pancreas:  Unremarkable 

**TUMOUR LIST TO FILL (input / template)**  
tumor id = tumor 3; organ = spleen; location = u; size = 18.0 mm; attenuation = hypoattenuating; series = se; image = im;

**FILLED LIST (expected model output)** 
tumor id = tumor 3; organ = spleen; location = u; size = 18.0 mm; attenuation = hypoattenuating; series = u; image = u;

**JUSTIFICATION**
tumor id = tumor 3; series = u, image = u — The rules ask me to check the tumor organ and size when matching tumors in the TUMOUR LIST TO FILL and in FINDINGS. tumor 3 in the list is a spleen tumor measuring 18mm. The FINDINGS mentions a spleen tumor with 18 mm, but does not have series number or image number for it: 'Spleen:  Similar-appearing hypoattenuating lesions, some of which are too small to characterize, which have been gradually increasing in size since at least 9/13/2021. The largest measures up to 1.8 cm.'. Therefore, I wrote series = u; image = u for tumor 3.

---

## YOUR ACTUAL TASK

Here is the FINDINGS section of the report to analyze:  
%(findings_section)s

Here is the TUMOR LIST to fill (replace “se” and “im”):  
%(tumors_list)s

After filling the list, provide a short justification for each tumor. The justification must explain the matching you did, and where in the findings you found the tumor organ and tumor size for each tumor in my list.
Quote the sentence(s) where you found the numbers, or explain why they are unknown.
"""



RefineNormalsKangPancreas="""
Core Instructions
You are a medical AI assistant tasked with analyzing CT radiology reports to identify suitable negative control cases for a pancreatic tumor detection AI validation study. Your role is to determine if a case should be EXCLUDED from the negative control cohort based on specific medical criteria.

TASK: Analyze the provided radiology report and determine if this case should be EXCLUDED from a negative control cohort for pancreatic AI tumor detection validation.

RESPOND WITH: 
- DECISION: EXCLUDE or INCLUDE
- REASONING: Brief explanation of key findings that led to your decision
- CONFIDENCE: High/Medium/Low

Comprehensive Exclusion Criteria
EXCLUDE this case if ANY of the following are present (1 to 7):
1. Pancreatic Abnormalities
- Any pancreatic mass, tumor, lesion, or nodule (any size)
- Pancreatic cysts, pseudocysts, or fluid collections
- Pancreatitis (acute, chronic, or autoimmune)
- Pancreatic duct dilatation (main duct >3mm in head, >2mm in body/tail)
- Common bile duct dilatation (>7mm)
- Pancreatic atrophy or focal attenuation differences
- Heterogeneous pancreatic enhancement patterns
- Intraductal papillary mucinous neoplasms (IPMNs)
- Mucinous cystic neoplasms or serous cystadenomas
- Pancreatic intraepithelial neoplasia
- Peripancreatic fat stranding (without acute pancreatitis)
- Duct cutoff or abrupt caliber change
- Intraductal filling defects or stones

Keywords in common in cases to EXCLUDE: "pancreatic mass", "pancreatic lesion", "pancreatic tumor", "IPMN", "pancreatic cyst", "pancreatic nodule", "focal pancreatic", "pancreatic abnormality", "duct dilatation", "pancreatic enhancement"

2. Prior Pancreatic Surgery History
- Pancreaticoduodenectomy (Whipple procedure)
- Distal pancreatectomy or central pancreatectomy
- Pancreatic enucleation
- Lateral pancreaticojejunostomy (Puestow procedure)
- Any pancreatic drainage procedures
- Pancreatic biopsy or fine needle aspiration
- Post-surgical pancreatic anastomosis
- Surgical clips in pancreatic region
- Pancreatic anastomotic sites
- Post-operative fluid collections in pancreatic bed
- Altered pancreatic ductal anatomy post-surgery
- Gastrojejunal or pancreaticojejunal anastomosis

Keywords in common in cases to EXCLUDE: "post-surgical", "pancreaticoduodenectomy", "Whipple", "distal pancreatectomy", "pancreatic resection", "pancreaticojejunostomy", "surgical clips", "post-operative changes", "s/p pancreatic", "prior pancreatic surgery", "pancreatic anastomosis", "surgical bed"

3. Acute Central Abdominal Findings
- Acute pancreatitis (any grade or severity)
- Acute cholangitis or ascending cholangitis
- Acute cholecystitis with pericholecystic fluid or wall thickening
- Acute diverticulitis with inflammatory changes
- Acute appendicitis with periappendiceal fluid
- Bowel obstruction with dilated loops and fluid levels
- Acute mesenteric ischemia or infarction
- Acute aortic pathology (dissection, rupture, aneurysm)
- Portal vein thrombosis or superior mesenteric vein thrombosis
- Splenic infarction or rupture
- Acute hemoperitoneum or retroperitoneal hematoma
- Acute ascites (new onset or rapidly accumulating)
- Acute pancreatic or peripancreatic fluid collections
- Post-procedural acute changes within 30 days

Keywords in common in cases to EXCLUDE: "acute pancreatitis", "acute cholangitis", "bowel obstruction", "acute abdomen", "hemoperitoneum", "acute inflammation", "portal vein thrombosis", "acute ascites", "acute mesenteric", "acute aortic"

4. Metastatic Disease
- Enlarged Lymph Nodes
- Liver or bone lesions
- Peritoneal carcinomatosis or peritoneal nodules
- Omental or mesenteric implants
- Liver metastases (any size or number)
- Lymphadenopathy >1.0cm in pancreaticoduodenal, celiac, or para-aortic regions
- Adrenal metastases
- Pelvic metastases or lymphadenopathy
- Indeterminate liver lesions requiring follow-up
- Borderline lymph nodes (0.8-1.0cm) in drainage regions
- Suspicious peritoneal thickening or enhancement
- Unexplained ascites

Keywords in common in cases to EXCLUDE: "metastases", "peritoneal carcinomatosis", "lymphadenopathy", "suspicious lesions", "peritoneal nodules", "omental implants", "liver metastases", "indeterminate liver lesions"

5. Adjacent Organ Pathologies
- Duodenal masses, ulcers, or wall thickening
- Gastric masses or severe gastritis
- Fundal varices
- Biliary obstruction or choledocholithiasis
- Cholangitis or biliary strictures
- Hepatic masses, cirrhosis, or significant steatosis
- Splenic masses or splenomegaly
- Superior mesenteric artery/vein abnormalities
- Significant ascites that may obscure pancreatic visualization

Keywords in common in cases to EXCLUDE: "duodenal mass", "gastric mass", "biliary obstruction", "bile duct dilatation", "choledocholithiasis", "hepatic mass", "splenic mass", "splenomegaly"

6. Technical and Quality Confounders
- Poor image quality or significant artifacts
- Inadequate contrast enhancement
- Motion artifacts affecting pancreatic visualization
- Slice thickness >3mm (if specified)
- Technical limitations affecting pancreatic assessment
- Insufficient contrast timing for pancreatic evaluation

Keywords in common in cases to EXCLUDE: "poor image quality", "motion artifact", "inadequate enhancement", "technical limitations", "suboptimal study", "limited evaluation"

7. Systemic Exclusions
- Current or recent (<5 years) history of any malignancy
- Suspiscious lesion in any organ
- Active inflammatory bowel disease with acute flares
- Autoimmune conditions affecting abdomen
- Patients receiving active cancer treatment

Keywords in common in cases to EXCLUDE: "history of cancer", "active malignancy", "inflammatory bowel disease", "cancer treatment", "oncology follow-up"

Analysis Framework
Step 1: Systematic Report Review
Scan the entire report including:
- Clinical history and indication
- Technique and contrast information  
- Findings section (organ by organ)
- Impression and recommendations
- Comparison with prior studies

Step 2: Sentence-Level Analysis
For each sentence, identify mentions of:
- Pancreas and peripancreatic structures
- Central abdominal organs (liver, spleen, kidneys, GI tract)
- Vascular structures (portal system, aorta, SMA, celiac axis)
- Peritoneal cavity and lymph node stations
- Surgical history or post-operative changes
- Acute findings or emergency conditions

Step 3: Finding Classification
Classify each relevant finding as:
- NORMAL: Explicitly stated as normal/unremarkable
- ABNORMAL: Any pathology requiring exclusion per criteria
- INDETERMINATE: Unclear or requires human review

Step 4: Decision Logic
IF any ABNORMAL findings match exclusion criteria → EXCLUDE
IF all pancreatic findings are NORMAL and no exclusion criteria met → INCLUDE  
IF INDETERMINATE findings present → Flag for human review with LOW confidence

Key Language Patterns
Inclusion Indicators (Required for INCLUDE decision):
- "normal pancreas", "unremarkable pancreas", "pancreas appears normal"
- "no pancreatic abnormalities", "no focal pancreatic lesions"
- "intact pancreatic architecture", "homogeneous pancreatic enhancement"
- "normal pancreatic duct", "no pancreatic mass"

Critical Exclusion Indicators:
- Any form of: "mass", "lesion", "tumor", "nodule", "cyst" + "pancreas/pancreatic"
- "pancreatitis", "duct dilatation", "IPMN", "pseudocyst"
- "post-surgical", "s/p", "status post", "surgical clips", "anastomosis"
- "acute", "obstruction", "thrombosis", "hematoma", "ascites"
- "metastases", "carcinomatosis", "lymphadenopathy", "implants"

Output Format
DECISION: [EXCLUDE/INCLUDE]
CONFIDENCE: [High/Medium/Low]

SYSTEMATIC FINDINGS:
- Pancreas: [Normal/Abnormal - specify findings]
- Surgical History: [Present/Absent - specify type if present]
- Acute Findings: [Present/Absent - specify if present]
- Metastatic Disease: [Present/Absent - specify if present]
- Adjacent Organs: [Normal/Abnormal - specify if abnormal]
- Technical Quality: [Adequate/Poor - specify if poor]

REASONING: [2-3 sentence explanation of decision rationale]

EXCLUSION CRITERIA MET: [List specific criteria numbers if EXCLUDE decision]

HUMAN REVIEW REQUIRED: [Yes/No - Yes if confidence is Low or indeterminate findings]

Quality Control Instructions
- If uncertain about any finding, choose EXCLUDE and flag for human review
- Prioritize patient safety - err on side of exclusion for unclear cases
- Pay special attention to surgical history which may be mentioned in clinical history
- Consider temporal relationships - acute findings take precedence
- Maintain high standards for negative controls to ensure AI validation integrity

"""


RefineNormalsPancreasCancer="""
### Instructions
1. Read the full report (plain text), paying special attention to clinical history and impressions.  
2. If you detect **any** of the following, set `DECISION: EXCLUDE`  
   • Prior history of cancer (e.g. “history of breast cancer”).  
   • Imaging evidence or impression of current cancer, metastasis, or tumor thrombus.  
   • Any lesion, nodule, mass, hyper/hypo/iso-attenuating focus, LI-RADS ≥ 3, or any suspicious lesion.  
3. Otherwise set `DECISION: INCLUDE`.  
4. Fill the template below and output nothing else—no extra text, no blank lines.

### Output template  (must match exactly)

- DECISION: <EXCLUDE or INCLUDE>
- JUSTIFICATION: <a complete justification for your decision, quoting the key report sentence(s) that triggered your decision>

### Example (should be excluded)

[START_REPORT]  
… CLINICAL HISTORY: 49-year-old female with **history of breast cancer** …  
IMPRESSION: New hypermetabolic right axillary node, recommend biopsy.  
[END_REPORT]

**Expected model output**

decision: EXCLUDE  
history_or_plausible_tumor: yes  
justification: Prior breast cancer noted and new hypermetabolic lymph node described as suspicious for malignancy.

### Now classify the next report
"""


multi_tumor_pathology_multi = """I am sending you below many pathology reports, all for the same patient.

YOUR TASK
Read ALL reports carefully and produce SIX lists:
1. organs_with_primary_malignant_tumors
2. organs_with_metastatic_tumors
3. organs_with_benign_tumors
4. organs_with_unknown_tumor_type
5. malignant_tumor_types
6. benign_tumor_types

The “tumor types” lists should specify, for each organ, the type of tumor identified (e.g., adenocarcinoma, squamous cell carcinoma, renal cell carcinoma, lipoma, hemangioma, cyst, abscess). If an organ has multiple types, list all separated by semicolons.

Then provide a justification for each organ listed, citing exact evidence phrases (quoted).

---

### OUTPUT FORMAT (strict; no extra prose before/after)

organs_with_primary_malignant_tumors: [<comma-separated organs>]
organs_with_metastatic_tumors: [<comma-separated organs>]
organs_with_benign_tumors: [<comma-separated organs>]
organs_with_unknown_tumor_type: [<comma-separated organs>]
malignant_tumor_types: {<organ>: <tumor type(s)>, ...}
benign_tumor_types: {<organ>: <tumor type(s)>, ...}

Justification:
- <organ>: <label> — Evidence: "<verbatim phrase>". Rationale: <1–4 lines>.
- ...

---

### DEFINITIONS (be conservative)

**Primary malignant tumor** = malignant neoplasm originating in that organ.  
If a report or history states a prior primary (even if “no residual tumor”), include that organ as primary and note “history only”.

**Metastatic tumor** = malignant tumor found in an organ but arising elsewhere (e.g., “metastatic carcinoma consistent with lung primary”).  

**Benign tumor/lesion** = explicitly benign lesion (e.g., “benign cyst”, “lipoma”, “adenoma”, “hemangioma”, “granuloma”, “abscess”, “fibroadenoma”, “leiomyoma”) OR a named mass/nodule/cyst/polyp described as benign.  

---

### BENIGN INCLUSION RULE (must meet BOTH)
A) Mentions a **lesion noun**: {“lesion”, “mass”, “nodule”, “tumor”, “neoplasm”, “cyst”, “polyp”, “adenoma”, “lipoma”, “hemangioma”, “leiomyoma”, “hamartoma”, “abscess”, “granuloma”}.  
B) The lesion is described as benign or is canonically benign.

### BENIGN EXCLUSION RULE
Do NOT include normal or uninvolved tissues:
- “Benign <normal tissue>” (e.g., “benign skeletal muscle”, “benign submandibular gland”) → exclude.
- “No tumor identified”, “negative for malignancy”, “benign mucosa/fat/muscle/lymph node” → exclude.
- Reactive or inflammatory processes (e.g., esophagitis, fibrosis, metaplasia, cholecystitis) without a mass → exclude.
- Fungal or bacterial infection without a lesion noun → exclude.

---

### UNKNOWN BUCKET
Use UNKNOWN for:
- “Atypical”, “suspicious”, “cannot rule out”, “insufficient for diagnosis”, or contradictory results across reports.

---

### CANONICAL ORGAN NAMES
Use exactly these names when applicable:  
uterus, prostate, right kidney, left kidney, right adrenal gland, left adrenal gland, pancreas, liver, spleen, bladder, gallbladder, duodenum, stomach, colon, esophagus.  
If the tumor is inside one of these, use the canonical name (e.g., use *uterus* for endometrial carcinoma).  
For others (e.g., lung, skin, lymph node), use the report’s term.

---

### EVIDENCE & JUSTIFICATION
- For EACH organ listed, provide one quoted phrase that proves the classification.  
- If classification is based on history, quote that phrase and mark “history only”.  
- Include specimen/part/date if available.  
- If the same organ has malignant and benign lesions, include it in both lists and explain.

---

### QUALITY CHECKS
- Prefer UNKNOWN over guessing.  
- Do NOT treat normal tissue or “benign <tissue>” as a benign tumor.  
- Be explicit about tumor types; if a lesion type is uncertain, label it as “unknown type” and explain.  
- Justify every entry with a phrase containing a lesion noun and diagnostic qualifier (benign/malignant/metastatic).

---

Now read the reports below and output ONLY the specified blocks.

Reports:
"""

multi_tumor_pathology="""You read ONE pathology report and output STRICT JSON only (no prose).

PRIORITY: Do NOT miss malignant tumors. Reports may contain BOTH malignant and benign findings; extract both independently.

############ HOW TO READ (SHORT, TWO-PASS) ############
Pass 1 — MALIGNANCY FIRST
- Look in this order and prefer later items only if earlier ones are absent/ambiguous:
  1) Final Pathologic Diagnosis
  2) Synoptic/Diagnostic Summary
  3) Addendum (finalized)
  4) Microscopic Diagnosis
  5) Gross Description
  6) Clinical History
- Positive (malignancy) cues: carcinoma, sarcoma, lymphoma, melanoma, “metastatic”, “invasive”, “malignant”, “high-grade dysplasia/CIS”, “carcinoma in situ”.
- Negative cues: “negative for malignancy”, “no evidence of carcinoma/malignancy”, “benign”, "polyp".
- “History of…/status post…” alone does NOT mean current malignancy in this specimen.
- In case of uncertain malignancy, answer 'u', do not answer 'yes'.
- In-situ disease (e.g., CIS) counts as malignant.
- Local extension ≠ metastasis (see below).

Decide:
- patient_has_malignant_tumor = true if any malignant diagnosis applies to the current specimen.
- false only if the report explicitly rules out malignancy for this specimen.
- "unknown" only if undecidable.

Extract malignant organs:
- Definition: malignant_organs = ALL organs with malignant involvement in this report (primary OR metastatic). If an organ appears in metastasis_organs, it MUST also appear in malignant_organs.
- Use this enum only: ["liver","kidney","pancreas","bladder","gallbladder","esophagus","stomach","duodenum","colon","prostate","uterus","spleen","adrenal gland","other"].
- Map terms: hepatic→liver; renal/renal pelvis/collecting system/pelvicalyceal→kidney; gastric→stomach; colonic/rectal→colon (treat rectum as colon here); urothelial→choose site (bladder or kidney pelvis). If a site isn't in the enum (e.g., ureter, bile duct, lung, bone, peritoneum), use "other" and quote the exact site.
- Laterality: "left"/"right"/"bilateral" if present; otherwise "unspecified". (GI/liver/bladder/gallbladder usually “unspecified” unless explicitly stated.)

Pass 2 — METASTASIS & BENIGN MASS/CYST
- Definition: metastasis_organs = ALL target organs where tumor has spread TO (e.g., “metastatic carcinoma to liver” → organ "liver"). If target not in enum, use "other" and quote the site (e.g., “lung”, “bone”, “lymph node”, “peritoneum”).
- metastasis_organs is a SUBSET of malignant_organs (metastatic targets only).
- Do NOT count local invasion as metastasis (e.g., “invades perirenal fat/renal sinus/pelvicalyceal system”, “serosal involvement”, “direct invasion to adjacent organ”). LVI alone ≠ metastasis.
- benign_mass_or_cyst: array of objects capturing ONLY benign masses or cysts.
  • Each object: { "organ": <enum>, "finding": <short phrase>, "rationale": <≤1 sentence>, "citations": [<short quotes>] }
  • Include only named benign masses (e.g., adenoma, oncocytoma, angiomyolipoma, lipoma, leiomyoma, hemangioma) or cysts (e.g., simple cyst, complex/hemorrhagic/mucinous cyst).
  • Exclude non-mass parenchymal changes (e.g., fibrosis, inflammation, arteriosclerosis), vascular changes, and generic normal statements.

Tumor types (normalize):
- Use these labels when matched (prefer the most specific):
  • “clear cell renal cell carcinoma” → "clear cell RCC"
  • “renal cell carcinoma” → "RCC"
  • “hepatocellular carcinoma” → "HCC"
  • “cholangiocarcinoma” → "CCA"
  • “pancreatic ductal adenocarcinoma” → "PDAC"
  • “pancreatic neuroendocrine tumor/carcinoma” → "PNET" (use "pNEC" only if “neuroendocrine carcinoma”)
  • “urothelial carcinoma” → "UC"
  • “adenocarcinoma” (site-agnostic) → "AdenoCA"
  • “squamous cell carcinoma” → "SCC"
  • “gastrointestinal stromal tumor” → "GIST"
  • “adrenocortical carcinoma” → "ACC"
- If malignant histology not in list, output a short raw name (≤60 chars). If only in-situ disease, append “ (in situ)”.

CITATIONS & RATIONALE (keep short)
- Every listed item must include a ≤2-sentence rationale and 1-2 short quotes (≤25 words) from the strongest section available.

SELF-CHECK (RECALL GUARD)
- Before finalizing, re-scan the note for malignant keywords (carcinoma, sarcoma, lymphoma, melanoma, metastatic, malignant, in situ). If any present and not contradicted by “negative for malignancy”, ensure patient_has_malignant_tumor=true and at least one malignant organ is listed (use "other" with a site quote if organ is unclear).
- If metastasis_organs is non-empty, set patient_has_malignant_tumor=true AND ensure every metastasis organ is also listed in malignant_organs.

############ STRICT JSON OUTPUT (KEY ORDER) ############
{
  "patient_has_malignant_tumor": {"value": true|false|"unknown", "rationale": string, "citations": [string, ...]},
  "malignant_organs": [ {"organ": string, "side": "left"|"right"|"bilateral"|"unspecified", "rationale": string, "citations": [string, ...]} ],
  "metastasis_organs": [ {"organ": string, "rationale": string, "citations": [string, ...]} ],
  "benign_mass_or_cyst": [ {"organ": string, "finding": string, "rationale": string, "citations": [string, ...]} ],
  "tumor_types": [string, ...]
}
ONLY return valid JSON; no extra keys/markdown.

############ MINI EXAMPLE (ONE-SHOT) ############
INPUT (snippet):
"FINAL PATHOLOGIC DIAGNOSIS: Left kidney—clear cell renal cell carcinoma, 6.2 cm. Margins negative. Addendum: Simple renal cyst in uninvolved parenchyma. No lymph nodes submitted."

OUTPUT:
{
  "patient_has_malignant_tumor": {
    "value": true,
    "rationale": "Final diagnosis confirms malignancy in current specimen.",
    "citations": ["Left kidney—clear cell renal cell carcinoma, 6.2 cm"]
  },
  "malignant_organs": [
    {
      "organ": "kidney",
      "side": "left",
      "rationale": "Cancer located in left kidney.",
      "citations": ["Left kidney—clear cell renal cell carcinoma"]
    }
  ],
  "metastasis_organs": [],
  "benign_mass_or_cyst": [
    {
      "organ": "kidney",
      "finding": "simple renal cyst",
      "rationale": "Benign cyst documented in addendum.",
      "citations": ["Addendum: Simple renal cyst"]
    }
  ],
  "tumor_types": ["clear cell RCC"]
}

############# TRUE INPUT FOR YOUR TASK #############
Pathology Report:

"""


def build_tumor_template(
    df: pd.DataFrame,
    accession: str,
    *,
    id_col: str = "Tumor ID",
    organ_col: str = "Organ",
    loc_col: str = "Tumor Location",
    size_col: str = "Tumor Size (mm)",
    att_col: str = "Tumor Attenuation",
    accn_col: str = "Encrypted Accession Number",
) -> str:
    """
    Return a prompt-ready tumour list (template) for one accession.

    Each line looks like:
        tumor id = 3; organ = liver; location = segment 7; size = 27 x 30 mm;
        attenuation = hypoattenuating; series = se; image = im;

    Missing / NaN values are replaced with 'u'.
    If a numeric size is present but lacks the unit, ' mm' is appended.
    """
    # 1) isolate the rows for this study
    rows = df.loc[df[accn_col] == accession].copy()

    if rows.empty:
        raise ValueError(f"No rows found for accession '{accession}'")

    # 2) helper to normalise values
    def _norm(value):
        if pd.isna(value) or str(value).strip() == "":
            return "u"
        return str(value).strip()

    def _size_with_unit(val: str) -> str:
        val = _norm(val)
        if val == "u":
            return val
        # if there is at least one digit and no 'mm' / 'cm', append ' mm'
        if re.search(r"\d", val) and not re.search(r"\b(?:mm|cm)\b", val, flags=re.I):
            return f"{val} mm"
        return val

    # 3) build the template lines
    lines = []
    for _, row in rows.sort_values(id_col).iterrows():
        line = (
            f"tumor id = {_norm(row[id_col])}; "
            f"organ = {_norm(row[organ_col])}; "
            f"location = {_norm(row[loc_col])}; "
            f"size = {_size_with_unit(row[size_col])}; "
            f"attenuation = {_norm(row[att_col])}; "
            f"series = se; image = im;"
        )
        lines.append(line)

    return "\n".join(lines)

#tumor id 1: organ = org; series number = se; image number = im \n---you fill the organ
def get_longitudinal_reports(df, patient_id):
    """
    For a given patient ID, extract all exam reports (from the 'Findings' column) and order them by exam date.
    Returns:
      - A list of ordered Encrypted Accession Numbers.
      - A list of strings, where each string is formatted as:
            Report {i} date: month/year
            Report {i} text: findings
        with i starting at 1.
    """
    # Filter the DataFrame to include only rows for the specified patient ID.
    patient_df = df[df["Patient ID"] == patient_id].copy()

    # Ensure 'Exam Completed Date' is in datetime format, then sort the rows by this date.
    patient_df["Exam Completed Date"] = pd.to_datetime(patient_df["Exam Completed Date"])
    patient_df = patient_df.sort_values("Exam Completed Date")

    # Extract the ordered list of Encrypted Accession Numbers.
    accession_numbers = patient_df["Encrypted Accession Number"].tolist()

    # Create the list of formatted report strings.
    report_strings = []
    for i, (_, row) in enumerate(patient_df.iterrows(), start=1):
        # Format the exam date as month/year.
        date_str = row["Exam Completed Date"].strftime("%m/%Y")
        findings = row["Findings"]
        report_str = f"Report {i} date: {date_str}\nReport {i} text: {findings}"
        report_strings.append(report_str)

    return accession_numbers, report_strings

def get_report_n_label(data,i,row_name='Anon Report Text',get_date=False,id_col='Accession Number',
                       get_2_reports=False,report2_col='radiology_Findings'):
    print(i)
    if isinstance(i,str):
        # get row with accession number i
        row=data[data[id_col]==i].to_dict('records')[0]
    else:
        row=data.iloc[i]
    if isinstance(row[row_name],str):
        report=row[row_name]
    else:
        report=None
        
    if get_2_reports:
        report2=row[report2_col]
        #raise ValueError('Do not forget to order the df by similarity before processing')
        return report, report2

    print(row)

    if get_date:
        
        print(row['Exam Started Date'])
        try:
            # Try parsing it with date and time
            date=row['Exam Started Date'].split()[0]
        except ValueError:
            # If it's just a date, return the string itself
            date=row['Exam Started Date']
        return report,date
    
    try:
        if not math.isnan(row['Liver Tumor']):
            label=f"liver tumor={int(row['Liver Tumor'])}; kidney tumor={int(row['Kidney Tumor'])}; pancreas tumor={int(row['Pancreas Tumor'])}"
        else:
            label=None
    except:
        label=None
    return report,label


risk_factors_pancreas = """
You are a clinical NLP assistant. Read the MEDICAL NOTE below and extract pancreatic cancer risk factors.
Return two parts ONLY, in this exact order:

1) A single-line JSON object between <BEGIN_JSON> and <END_JSON>.
2) A justification section between <BEGIN_JUSTIFICATION> and <END_JUSTIFICATION>.

GENERAL RULES: determine if each risk factor is positive (yes), negative (no), or not mentioned in the note (absent).
- Only consider what is explicitly written, do not infer anything beyond what is written. For example, if the note mentions pancreas atrophy but it does not EXPLICITLY mention pancreatitis, you CANNOT infer pacreatis.
- Use "yes" only if the note clearly supports presence (current or past).
- Use "no" only if the note explicitly denies the risk factor or states normal/negative findings.
- If the note provides no information about a risk factor, use "absent".
- Categorical answers MUST be one of: "yes", "no", or "absent" (lowercase).
- Height/weight: return numeric values (converted), or null if not found.
- In the justification, quote short, verbatim snippets from the note to justify each "yes" and each explicit "no" (e.g., "denies smoking").
- Preserve measurements, units, dates as written when quoting evidence.
- Carefully read all parts of the note.
- Consider risk factors as positive ("yes") both if they are current or past (e.g., "history of smoking" counts as "yes").

RISK FACTOR DEFINITIONS & DECISION LOGIC
1) smoking: "yes" if current OR former smoker is stated. "no" if never/denies. If absent → "absent".
   • Synonyms: smoker, tobacco, cigarettes, pack-year(s), quit smoking,...
2) obesity: "yes" if text explicitly says obese/morbidly obese/overweight. "no" if explicitly denied. If absent → "absent".
3) patient_weight_kg: Extract weight. Accept kg or lb. Convert lb→kg (kg = lb * 0.45). One decimal.
   • Prefer the most recent/current value if multiple are present.
4) patient_height_cm: Extract height. Accept cm, m, ft/in. Convert to cm. One decimal.
   • Examples: 5'9", 5 ft 9 in, 175 cm, 1.75 m.
5) alcohol_use: "yes" if current alcohol use or documented alcohol abuse/dependence (current/past). "no" if denies/none. If absent → "absent".
6) chronic_pancreatitis: "yes" if explicitly stated (including history of). "no" if explicitly denied. Otherwise "absent".
7) acute_pancreatitis: "yes" if explicitly stated. "no" if explicitly denied. Otherwise "absent".
8) pancreatitis: "yes" if either chronic or acute pancreatitis is stated (or unclear which). "no" if both are explicitly denied. Otherwise "absent".
9) family_history_pancreatic_cancer: "yes" if family history is stated. "no" if explicitly denied. Otherwise "absent".
10) high_risk_germline_mutation: "yes" if any of: BRCA1, BRCA2, PALB2, ATM, CDKN2A, STK11, Lynch (MLH1/MSH2/MSH6/PMS2/EPCAM). "no" if explicitly negative. Otherwise "absent".
   • Also output "high_risk_mutation_list": list of any genes/syndromes found (strings), empty list if none.
11) diabetes: "yes" if diabetes is indicated. "no" if explicitly denied or ruled out. Otherwise "absent".
12) weight_loss: "yes" if unintentional weight loss is mentioned. "no" if explicitly denied. Otherwise "absent".
13) painless_jaundice: "yes" if 'painless jaundice' or jaundice without pain is stated. "no" if jaundice explicitly denied. Otherwise "absent".
14) pruritus: "yes" if pruritus/itching is mentioned. "no" if explicitly denied. Otherwise "absent".
15) dark_urine: "yes" if dark urine/tea-colored urine is mentioned. "no" if explicitly denied. Otherwise "absent".
16) pale_acholic_stools: "yes" if pale/acholic stools are mentioned. "no" if explicitly denied. Otherwise "absent".
17) epigastric_pain: "yes" if mentioned. "no" if explicitly denied. Otherwise "absent".
18) back_pain: "yes" if mentioned. "no" if explicitly denied. Otherwise "absent".
19) anorexia: "yes" if mentioned. "no" if explicitly denied. Otherwise "absent".
20) steatorrhea: "yes" if mentioned. "no" if explicitly denied. Otherwise "absent".
21) dvt_or_pe: "yes" if DVT or PE is stated (history or current). "no" if explicitly denied. Otherwise "absent".
22) abdominal_pain: "yes" if stated. "no" if explicitly denied. Otherwise "absent".
23) high_direct_bilirubin_or_alp: "yes" if the note flags direct bilirubin or ALP as elevated/high/above reference range, OR provides a value above the stated ULN.
   • If values present with explicit normal ranges → judge against that range.
   • If only numbers without ranges and no “high” flag → "absent".
   • Also return "direct_bilirubin_value", "direct_bilirubin_unit", "alp_value", "alp_unit" if present; else null.
24) elevated_ca19_9: "yes" if flagged as elevated/high/above range OR numeric value > ULN.
   • If no range is given, use 35 U/mL as conventional ULN.
   • Also return "ca19_9_value" (number), "ca19_9_unit" (string) if present; else null.
25) high_nlr_or_thrombocytosis_or_low_albumin: "yes" if ANY is flagged or numerically outside range:
   • Prefer note's ranges/flags. If none:
     - NLR > 3 (use if both ANC and ALC are present or NLR given).
     - Thrombocytosis: platelets > 400 x10^9/L (aka > 400k/µL).
     - Low albumin: albumin < 3.5 g/dL.
   • "no" if values explicitly normal/below thresholds. Otherwise "absent".
   • Also return any of: "nlr", "anc", "alc", "platelets", "albumin" with units/values if present; else null.
26) pancreatic_duct_dilatation: "yes" if MPD/pancreatic duct is described as dilated. "no" if explicitly denied. Otherwise "absent".
27) pancreatic_focal_atrophy_or_contour_change: "yes" if focal atrophy or contour change is described. "no" if explicitly denied. Otherwise "absent".
28) ipmn: "yes" if intraductal papillary mucinous neoplasm (IPMN) is stated. "no" if explicitly denied. Otherwise "absent".
29) pancreatic_cyst: "yes" if any pancreatic cyst is stated. "no" if explicitly denied. Otherwise "absent".

OUTPUT FORMAT (STRICT)
Print exactly:

<BEGIN_JSON>
{  ONE SINGLE-LINE JSON OBJECT HERE  }
<END_JSON>
<BEGIN_JUSTIFICATION>
- smoking: "..."  
- obesity: "..."
- patient_weight_kg: "..."  
- patient_height_cm: "..."
- alcohol_use: "..."
- chronic_pancreatitis: "..."
- acute_pancreatitis: "..."
- pancreatitis: "..."
- family_history_pancreatic_cancer: "..."
- high_risk_germline_mutation: "..."
- diabetes: "..."
- weight_loss: "..."
- painless_jaundice: "..."
- pruritus: "..."
- dark_urine: "..."
- pale_acholic_stools: "..."
- epigastric_pain: "..."
- back_pain: "..."
- anorexia: "..."
- steatorrhea: "..."
- dvt_or_pe: "..."
- abdominal_pain: "..."
- high_direct_bilirubin_or_alp: "..."  
- elevated_ca19_9: "..."               
- high_nlr_or_thrombocytosis_or_low_albumin: "..." 
- pancreatic_duct_dilatation: "..."
- pancreatic_focal_atrophy_or_contour_change: "..."
- ipmn: "..."
- pancreatic_cyst: "..."
<END_JUSTIFICATION>

JSON KEYS (use exactly these; lowercase snake_case):
{
  "smoking": "yes|no|absent",
  "obesity": "yes|no|absent",
  "patient_weight_kg": <number or null>,
  "patient_height_cm": <number or null>,
  "bmi": <number or null>,

  "alcohol_use": "yes|no|absent",
  "chronic_pancreatitis": "yes|no|absent",
  "acute_pancreatitis": "yes|no|absent",
  "pancreatitis": "yes|no|absent",
  "family_history_pancreatic_cancer": "yes|no|absent",

  "high_risk_germline_mutation": "yes|no|absent",
  "high_risk_mutation_list": [strings],

  "diabetes": "yes|no|absent",
  "weight_loss": "yes|no|absent",

  "painless_jaundice": "yes|no|absent",
  "pruritus": "yes|no|absent",
  "dark_urine": "yes|no|absent",
  "pale_acholic_stools": "yes|no|absent",

  "epigastric_pain": "yes|no|absent",
  "back_pain": "yes|no|absent",
  "anorexia": "yes|no|absent",
  "steatorrhea": "yes|no|absent",
  "dvt_or_pe": "yes|no|absent",
  "abdominal_pain": "yes|no|absent",

  "high_direct_bilirubin_or_alp": "yes|no|absent",
  "direct_bilirubin_value": <number or null>,
  "direct_bilirubin_unit": <string or null>,
  "alp_value": <number or null>,
  "alp_unit": <string or null>,

  "elevated_ca19_9": "yes|no|absent",
  "ca19_9_value": <number or null>,
  "ca19_9_unit": <string or null>,

  "high_nlr_or_thrombocytosis_or_low_albumin": "yes|no|absent",
  "nlr": <number or null>,
  "anc_value": <number or null>,
  "anc_unit": <string or null>,
  "alc_value": <number or null>,
  "alc_unit": <string or null>,
  "platelets_value": <number or null>,
  "platelets_unit": <string or null>,
  "albumin_value": <number or null>,
  "albumin_unit": <string or null>,

  "pancreatic_duct_dilatation": "yes|no|absent",
  "pancreatic_focal_atrophy_or_contour_change": "yes|no|absent",
  "ipmn": "yes|no|absent",
  "pancreatic_cyst": "yes|no|absent"
}

CONVERSIONS
- Height:
  • if meters (m): cm = m * 100
  • if feet/inches: cm = 2.54 * (12*feet + inches)
- Weight:
  • if pounds (lb): kg = lb * 0.45
- Round cm/kg/BMI to one decimal.

EXAMPLE (fabricated)
NOTE: 72-year-old male. Former smoker (30 pack-years), quit 2010. Drinks wine socially. Reports 10-lb unintentional weight loss. Height 5'10", Weight 190 lb. Abdominal pain present. No jaundice. Labs: CA 19-9 42 U/mL (ULN 35). Albumin 3.2 g/dL (low). Platelets 420k/µL (high). Imaging: main pancreatic duct mildly dilated to 4 mm; 1.5 cm pancreatic cyst in body. No pancreatitis history. No family history of pancreatic cancer. BRCA2 positive.

EXPECTED OUTPUT:
<BEGIN_JSON>
{"smoking":"yes","obesity":"absent","patient_weight_kg":86.2,"patient_height_cm":177.8,"bmi":27.3,"alcohol_use":"yes","chronic_pancreatitis":"no","acute_pancreatitis":"no","pancreatitis":"no","family_history_pancreatic_cancer":"no","high_risk_germline_mutation":"yes","high_risk_mutation_list":["BRCA2"],"diabetes":"absent","weight_loss":"yes","painless_jaundice":"no","pruritus":"absent","dark_urine":"absent","pale_acholic_stools":"absent","epigastric_pain":"absent","back_pain":"absent","anorexia":"absent","steatorrhea":"absent","dvt_or_pe":"absent","abdominal_pain":"yes","high_direct_bilirubin_or_alp":"absent","direct_bilirubin_value":null,"direct_bilirubin_unit":null,"alp_value":null,"alp_unit":null,"elevated_ca19_9":"yes","ca19_9_value":42.0,"ca19_9_unit":"U/mL","high_nlr_or_thrombocytosis_or_low_albumin":"yes","nlr":null,"anc_value":null,"anc_unit":null,"alc_value":null,"alc_unit":null,"platelets_value":420000.0,"platelets_unit":"per µL","albumin_value":3.2,"albumin_unit":"g/dL","pancreatic_duct_dilatation":"yes","pancreatic_focal_atrophy_or_contour_change":"absent","ipmn":"absent","pancreatic_cyst":"yes"}
<END_JSON>
<BEGIN_JUSTIFICATION>
- smoking: "Former smoker (30 pack-years), quit 2010"
- obesity: (no mention of obesity/overweight)
- patient_weight_kg: "Weight 190 lb"
- patient_height_cm: "Height 5'10""
- alcohol_use: "Drinks wine socially"
- chronic_pancreatitis: "No pancreatitis history"
- acute_pancreatitis: "No pancreatitis history"
- pancreatitis: "No pancreatitis history"
- family_history_pancreatic_cancer: "No family history of pancreatic cancer"
- high_risk_germline_mutation: "BRCA2 positive"
- diabetes: (not mentioned)
- weight_loss: "Reports 10-lb unintentional weight loss"
- painless_jaundice: "No jaundice"
- pruritus: (not mentioned)
- dark_urine: (not mentioned)
- pale_acholic_stools: (not mentioned)
- epigastric_pain: (not mentioned)
- back_pain: (not mentioned)
- anorexia: (not mentioned)
- steatorrhea: (not mentioned)
- dvt_or_pe: (not mentioned)
- abdominal_pain: "Abdominal pain present"
- high_direct_bilirubin_or_alp: (no values/flags given)
- elevated_ca19_9: "CA 19-9 42 U/mL (ULN 35)"
- high_nlr_or_thrombocytosis_or_low_albumin: "Albumin 3.2 g/dL (low); Platelets 420k/µL (high)"
- pancreatic_duct_dilatation: "main pancreatic duct mildly dilated to 4 mm"
- pancreatic_focal_atrophy_or_contour_change: (not mentioned)
- ipmn: (not mentioned)
- pancreatic_cyst: "1.5 cm pancreatic cyst in body"
<END_JUSTIFICATION>

NOW PROCESS THIS NOTE:

"""

pancreatic_tumor_staging_prompt = """
You are given a radiology report. Extract pancreatic tumor findings and vascular encasement details into STRICT JSON following the schema and rules below. Follow the definitions carefully.

=====================
1. OVERALL TASK
=====================

- I will send you ONE radiology report at a time.
- Your job is to extract ALL tumors located in the pancreas (primary lesion and any additional pancreatic lesions, if present).
- Output MUST be a single JSON object that follows the schema in Section 2.
- You MUST also follow the encasement rules in Section 3, which distinguish solid tumor contact from hazy/stranding contact.

=====================
2. JSON SCHEMA
=====================

The output must be a single JSON object with the following top-level keys:

{
  "no_pancreatic_tumor": <bool>,
  "no_pancreatic_tumor_justification": <string>,
  "no_pancreatic_tumor_supporting_quotes": [<string>, ...],
  "tumors": [ ... ]
}

- "no_pancreatic_tumor": true if the report does NOT mention any pancreatic tumor; false otherwise.
- "no_pancreatic_tumor_justification":
  - If no_pancreatic_tumor is true: a short explanation of why you concluded there is no pancreatic tumor.
  - If no_pancreatic_tumor is false: use an empty string "".
- "no_pancreatic_tumor_supporting_quotes":
  - If no_pancreatic_tumor is true: array of 0-3 direct phrases from the report that support the absence of a pancreatic tumor (e.g., explicit normal pancreas, or no mention of masses).
  - If no_pancreatic_tumor is false: use [] (empty list).

- "tumors": an array with ONE ENTRY PER PANCREATIC TUMOR mentioned in the report.
  - If there is no pancreatic tumor, "tumors" must be [] (empty list).

Each element of "tumors" must be an object with the following keys:

{
  "tumor_index": <int>,
  "tumor_type": <string>,
  "tumor_type_other_detail": <string>,
  "type_certainty": <"low" | "medium" | "high">,
  "size_mm": [<number>, ...],
  "location": <"head" | "neck" | "body" | "tail" | "unknown">,
  "vessel_encasement": {
    "CA": <"encasement" | "no_encasement" | "unknown">,
    "CHA": <"encasement" | "no_encasement" | "unknown">,
    "SMA": <"encasement" | "no_encasement" | "unknown">,
    "SMV": <"encasement" | "no_encasement" | "unknown">,
    "MPV": <"encasement" | "no_encasement" | "unknown">,
    "aorta": <"encasement" | "no_encasement" | "unknown">
  },
  "justification": <string>,
  "supporting_quotes": {
    "tumor_type": [<string>, ...],
    "size": [<string>, ...],
    "location": [<string>, ...],
    "vessels": {
      "CA": [<string>, ...],
      "CHA": [<string>, ...],
      "SMA": [<string>, ...],
      "SMV": [<string>, ...],
      "MPV": [<string>, ...],
      "aorta": [<string>, ...]
    }
  }
}

Field definitions:

- "tumor_index": integer starting at 1 for this report (1, 2, 3, ...). Arbitrary.
- "tumor_type": one of:
  - "PDAC"
  - "PNET"
  - "cyst"
  - "unknown"
  - "other"
- "tumor_type_other_detail":
  - If tumor_type is "other": fill with the tumor type as written in the report (e.g., "acinar cell carcinoma", "solid pseudopapillary neoplasm").
  - If tumor_type is NOT "other": use an empty string "".
- "type_certainty": one of "low", "medium", "high" indicating how confident you are in the tumor type based on the text.
- "size_mm": array of 1-3 numbers in millimeters.
  - Convert any size from cm to mm (e.g., "4.6 x 3.3 cm" -> [46, 33]).
  - If no size is reported, use [] (empty list).
  - If multiple sizes are reported for a single tumor, we only care about the current tumor size, not prior sizes. 
- "location": map report text to:
  - "head" for pancreatic head, uncinate process, etc.
  - "neck" for pancreatic neck.
  - "body" for pancreatic body.
  - "tail" for pancreatic tail.
  - "unknown" if the location within the pancreas is not specified.
- "vessel_encasement": see Section 3 for detailed rules.
- "justification": 1-4 sentences explaining how you decided:
  - that this is a pancreatic tumor,
  - the tumor type and certainty,
  - the size and location,
  - and the encasement status for each vessel (important).
- "supporting_quotes":
  - "tumor_type": array of 0-3 direct phrases from the report supporting the tumor type.
  - "size": array of 0-3 direct phrases supporting the size.
  - "location": array of 0-3 direct phrases supporting the location.
  - "vessels": for each vessel key, array of 0-3 direct phrases supporting the encasement classification for that vessel.
  - If there are no relevant phrases, use [].

=====================
3. ENCASEMENT RULES
=====================

You MUST follow these rules exactly. 

Evaluate whether the tumor encases the celiac axis (CA), the superior mesenteric artery (SMA), the common hepatic artery (CHA), the superior mesenteric vein (SMV), the aorta, and the main portal vain (MPV). 

For each vessel (CA, CHA, SMA, SMV, MPV, aorta), classify as:
- "encasement"
- "no_encasement"
- "unknown"

A. When to label "encasement":
--------------------------------
A blood vessel is considered "encasement" if:

1) The report explicitly uses the term "encasement" or a clear synonym
   (e.g., "encases", "360-degree encasement", "circumferential involvement").
   - IMPORTANT: Generic "involvement" or "contact" alone does NOT automatically mean encasement.
   - You must look for wording that implies more than 180 degrees of solid contact.

OR

2) The report states that there is SOLID tumor contact with the vessel of MORE THAN 180 degrees
   (e.g., "solid tumor contact >180 degrees", ">180° solid contact").
   - Hazy attenuation or stranding contact is NOT the same as solid tumor contact and does NOT count for determining encasement.
   - If the report only mentions one type of contact and does not clearly define it as hazy attenuation/stranding contact, you MUST assume it is solid tumor contact.

B. When to label "no_encasement":
-----------------------------------
A blood vessel is considered "no_encasement" if:

1) The report explicitly states that there is no encasement, or uses a clear synonym indicating absence of encasement.

OR

2) The report clearly states that the tumor does NOT contact the vessel (no involvement).

OR

3) The report states that there is SOLID tumor contact of 180 degrees OR LESS
   (e.g., "abutment", "solid contact <=180 degrees", "=180 degrees").
   - IMPORTANT: Hazy attenuation/stranding contact alone does NOT count when deciding encasement versus no_encasement. A tumor with solid concact of <=180 degrees and hazy contact >180 degrees is still "no_encasement".

C. When to label "unknown":
----------------------------
Label encasement as "unknown" if:

1) The report explicitly says that encasement or vascular involvement cannot be determined.

OR

2) The report does NOT describe contact between the pancreatic tumor and that specific vessel.

OR

3) The report explicitly provides only hazy attenuation/stranding contact, without providing SOLID tumor contact angle. If only one type of contact is provided and it is not clearly defined as hazy/stranding, assume it is solid contact and do NOT label "unknown".

OR

4) The report does not mention a pancreatic tumor at all (in that case, all vessels must be "unknown" and no_pancreatic_tumor must be true).

=====================
4. TUMOR TYPE & CERTAINTY
=====================

- "PDAC":
  - Use when the report calls it "adenocarcinoma of the pancreas", "pancreatic ductal adenocarcinoma", "pancreatic cancer" with typical malignant features, or similar.
  - If the type is explicitly named as adenocarcinoma of the pancreas, set type_certainty = "high".
- "PNET":
  - Use when the report clearly mentions "neuroendocrine tumor", "PNET", etc.
- "cyst":
  - Use when the lesion is clearly described as a cystic lesion of the pancreas (e.g., "pancreatic cyst", "cystic lesion").
- "unknown":
  - Use when the report does not provide enough information to determine the type.
- "other":
  - Use when the tumor type is clearly defined but does not fit PDAC, PNET, or cyst.
  - In this case, set:
    - tumor_type = "other"
    - tumor_type_other_detail = the tumor type as written in the report (e.g., "acinar cell carcinoma").

Set "type_certainty" as:
- "high": type is explicitly named or very clearly implied.
- "medium": type is not explicitly stated but strongly supported by the pattern (e.g., classic PDAC pattern, pancreatic duct enlargement).
- "low": type is weakly suggested or ambiguous.

=====================
5. CASE WITH NO PANCREATIC TUMOR
=====================

If the report does NOT describe any pancreatic tumor:

- Set:
  - "no_pancreatic_tumor": true
  - "tumors": []
  - "no_pancreatic_tumor_justification": short explanation (1–3 sentences) of why you concluded there is no pancreatic tumor.
  - "no_pancreatic_tumor_supporting_quotes": array of 0–3 direct phrases supporting this conclusion (e.g., normal pancreas, or absence of mass description).

- EXAMPLE TEMPLATE TO COPY (then fill justification and quotes):

{
  "no_pancreatic_tumor": true,
  "no_pancreatic_tumor_justification": "There is no mention of a pancreatic mass or tumor, and the pancreas is described as normal.",
  "no_pancreatic_tumor_supporting_quotes": [
    "Pancreas: Unremarkable.",
    "No pancreatic mass is identified."
  ],
  "tumors": []
}

You should adapt the justification and supporting quotes to match the actual report.

=====================
6. EXAMPLES WITH PANCREATIC TUMORS
=====================

Below are two full example reports and their expected JSON outputs to guide your behavior.

-----------------------------------
EXAMPLE 1 — INPUT REPORT
-----------------------------------

CT ABDOMEN/PELVIS WITH AND WITHOUT CONTRAST  [DATE]:10 PM
CLINICAL HISTORY: Baseline, pancreatic cancer
COMPARISON:  Outside [ADDRESS] abdomen [DATE]
Techniques: Contiguous 5 mm collimation images were obtained through the abdomen without intravenous contrast. Subsequently, 1.25 mm axial images were acquired through the abdomen during the arterial phase followed by 1.25 mm axial images through the abdomen and pelvis during portal venous phase, and through the abdomen at 5 minutes delay. 
MEDICATIONS:
Iohexol 350 - 100 mL - Intravenous
Findings:
Pancreatic tumor:
1)  Location:  body
2)  Size:  4.6 x 3.3 cm (series 304 image 52), previously 4.0 x 3.0 cm
3)  Enhancement relative to pancreas: Hypoattenuating
4)  Biliary obstruction: no
5)  Pancreatic duct obstruction: yes
[PERSONALNAME]:
       1) Celiac axis is involved.
            -Solid tumor contact: >180
            -Focal vessel narrowing or contour irregularity: yes
       2)  SMA is involved. 
            -Solid tumor contact: <=180
            -Focal vessel narrowing or contour irregularity: no
            -Extension into first branching artery: no
       3)  CHA is involved. 
            -Solid tumor contact: >180
            -Focal vessel narrowing or contour irregularity: yes
            -Extension into celiac axis: yes 
            -Extension to bifurcation of right/left hepatic artery: no
       4)  Arterial variant: None
[PERSONALNAME]:
1)  MPV is involved. 
           -Solid tumor contact: =180
           -Focal vessel narrowing or contour irregularity: no
2)  SMV is involved. 
            -Solid tumor contact: =180
            -Focal vessel narrowing or contour irregularity: no
    -Extension into first draining vein: no
3)  Thrombus in vein:  absent
4)  Venous collateral:  absent
Liver: Multiple hypoattenuating lesions are new or increased in size, for example: 
-Segment 7 measuring 1.6 x 1.4 cm (series 304 image 34), previously 1.1 x 1.1 cm
-Segment 2 measuring 1.9 x 1.7 cm (series 304 image 40), previously 1.2 x 1.1 cm
-New 1.0 cm lesion in the caudate (series 304 image 37)
-New 0.7 cm lesion in segment 2 (series 304 image 33)
New geographic hypoattenuation of segment 6 likely secondary to new occlusive bland thrombus in the right hepatic vein (series 304 image 42) which is at least 4.5 cm from the inferior vena cava.
Decreased now mild intrahepatic and extrahepatic biliary ductal dilation. New pneumobilia compatible with interval sphincterotomy.
Gallbladder: Status post cholecystectomy.
Peritoneal or omental nodules:  None
Ascites: absent
Lymph nodes:  Similar small peripancreatic lymph nodes.
Spleen: Unremarkable
Adrenal: Unremarkable
Kidney: Unremarkable
GI tract: Unremarkable
Reproductive organs: Small amount of fluid in the endometrial canal.
Visualized lung bases:  For chest findings, please see the separately dictated report from the CT of the chest of the same date.
Bones:  No suspicious lesions
Extraperitoneal soft tissues: Unremarkable
Lines/drains/medical devices: None
RADIATION DOSE INDICATORS:
Exposure Events: 8 , CTDIvol Min: 0.1 mGy, CTDIvol Max: 19.3 mGy, DLP: 2507.4 mGy.cm

IMPRESSION: 
1.  Compared to [DATE], increased size of the pancreatic body mass which now encases the celiac axis and common hepatic artery and abuts the superior mesenteric artery and portosplenic confluence.
2.  New and enlarging hepatic metastases.
3.  New right hepatic vein bland thrombosis which is at least 4.5 cm from the inferior vena cava.
Report dictated by: [PERSONALNAME], MD, signed by: [PERSONALNAME] [ADDRESS], MD
Department of Radiology and Biomedical Imaging

-----------------------------------
EXAMPLE 1 — EXPECTED JSON OUTPUT
-----------------------------------

{
  "no_pancreatic_tumor": false,
  "no_pancreatic_tumor_justification": "",
  "no_pancreatic_tumor_supporting_quotes": [],
  "tumors": [
    {
      "tumor_index": 1,
      "tumor_type": "PDAC",
      "tumor_type_other_detail": "",
      "type_certainty": "high",
      "size_mm": [46, 33],
      "location": "body",
      "vessel_encasement": {
        "CA": "encasement",
        "CHA": "encasement",
        "SMA": "no_encasement",
        "SMV": "no_encasement",
        "MPV": "no_encasement",
        "aorta": "unknown"
      },
      "justification": "There is a pancreatic body mass in a patient with pancreatic cancer, hypoattenuating and causing pancreatic duct obstruction with hepatic metastases, consistent with PDAC. The celiac axis (CA) and common hepatic artery (CHA) have solid tumor contact >180 degrees and are described as encased. The SMA, MPV, and SMV have solid tumor contact equal to or less than 180 degrees (abutment), which does not meet the criteria for encasement. The aorta is not mentioned in relation to tumor contact, so its encasement status is unknown.",
      "supporting_quotes": {
        "tumor_type": [
          "CLINICAL HISTORY: Baseline, pancreatic cancer",
          "Enhancement relative to pancreas: Hypoattenuating",
          "New and enlarging hepatic metastases."
        ],
        "size": [
          "Size:  4.6 x 3.3 cm (series 304 image 52)"
        ],
        "location": [
          "Location:  body",
          "increased size of the pancreatic body mass"
        ],
        "vessels": {
          "CA": [
            "Celiac axis is involved.",
            "-Solid tumor contact: >180",
            "mass which now encases the celiac axis"
          ],
          "CHA": [
            "CHA is involved.",
            "-Solid tumor contact: >180",
            "mass which now encases the celiac axis and common hepatic artery"
          ],
          "SMA": [
            "SMA is involved.",
            "-Solid tumor contact: <=180",
            "abuts the superior mesenteric artery"
          ],
          "SMV": [
            "SMV is involved.",
            "-Solid tumor contact: =180"
          ],
          "MPV": [
            "MPV is involved.",
            "-Solid tumor contact: =180",
            "abuts the superior mesenteric artery and portosplenic confluence"
          ],
          "aorta": []
        }
      }
    }
  ]
}

-----------------------------------
EXAMPLE 2 — INPUT REPORT
-----------------------------------

For metastatic adenocarcinoma of the pancreas with liver metastases, according to RECIST 1.1 with progressive disease: - Progressive size of liver metastases and new liver lesions. - Progressive size of lymph node in the hepatic hilum. - Progressive size of pulmonary nodules. - Progressive dilatation of intrahepatic bile ducts and gallbladder hydrops compared to the previous examination. - Pre-existing occlusion of the superior mesenteric vein. 360° encasement of the superior mesenteric artery without caliber irregularity. - Pre-existing gastric diverticulum.  Telephone notification to the attending physician on duty at the intensive care unit on 06.04.2020 at 22:

-----------------------------------
EXAMPLE 2 — EXPECTED JSON OUTPUT
-----------------------------------

{
  "no_pancreatic_tumor": false,
  "no_pancreatic_tumor_justification": "",
  "no_pancreatic_tumor_supporting_quotes": [],
  "tumors": [
    {
      "tumor_index": 1,
      "tumor_type": "PDAC",
      "tumor_type_other_detail": "",
      "type_certainty": "high",
      "size_mm": [],
      "location": "unknown",
      "vessel_encasement": {
        "CA": "unknown",
        "CHA": "unknown",
        "SMA": "encasement",
        "SMV": "unknown",
        "MPV": "unknown",
        "aorta": "unknown"
      },
      "justification": "The report explicitly mentions metastatic adenocarcinoma of the pancreas, which corresponds to PDAC, but does not specify the size or the precise location within the pancreas. It describes 360-degree encasement of the superior mesenteric artery, which is by definition more than 180 degrees of solid contact and therefore encasement. The superior mesenteric vein is described as occluded, but there is no explicit description of solid contact, so its encasement status is considered unknown according to the given rules. Other vessels are not described in relation to tumor contact, so they are also classified as unknown.",
      "supporting_quotes": {
        "tumor_type": [
          "metastatic adenocarcinoma of the pancreas with liver metastases"
        ],
        "size": [],
        "location": [],
        "vessels": {
          "CA": [],
          "CHA": [],
          "SMA": [
            "360° encasement of the superior mesenteric artery without caliber irregularity."
          ],
          "SMV": [
            "Pre-existing occlusion of the superior mesenteric vein."
          ],
          "MPV": [],
          "aorta": []
        }
      }
    }
  ]
}

=====================
7. FINAL INSTRUCTION
=====================

Now, when I provide a NEW radiology report, return ONLY a single JSON object that follows the schema and rules above. Do not include any extra commentary or explanations outside of the JSON.
"""

def get_instuctions(fast,step,examples=0,organ='liver',all_data=None,accession=None):
    
    if step=='tumor detection':
        if fast:
            if len(examples)==0:
                instructions=instructions0ShotFast
            elif len(examples)==1:
                instructions=instructions1ShotFast
            else:
                instructions=instructionsFewShotFast % {'examples':len(examples),'last':len(examples)+1}
        else:
            if len(examples)==0:
                instructions=instructions0Shot
            elif len(examples)==1:
                instructions=instructions1Shot
            else:
                instructions=instructionsFewShot % {'examples':len(examples),'last':len(examples)+1}
    elif step=='tumor slices':
        #create the template to fill
        tumor_list = build_tumor_template(all_data, accession)
        filtered = all_data[all_data["Encrypted Accession Number"] == accession]
        if filtered.empty:
            raise ValueError(f"No rows found for accession={accession!r}")
        filtered = filtered.head(1)  # Get the first row that matches the accession number
        finding = filtered['Findings'].iloc[0]
        instructions = get_tumor_slices % {'findings_section': finding,
                                           'tumors_list':tumor_list}
    elif step=='pre-diagnostic confirmation':
        instructions=preDiagnositcConfirmation
    elif step=='find matching reports':
        instructions=compare_reports
    elif step=='malignancy detection':
        if len(examples)>0:
            raise ValueError('Only 0 or 1 examples allowed for malignancy detection')
        if not fast:
            if organ=='liver':
                instructions=instructions0ShotMalignancy % {'organ':organ,
                                                        'benign_tumors':benign_liver,
                                                        'malignant_tumors':malignant_liver,
                                                        'both_tumors':both_liver}
            elif organ=='pancreas':
                instructions=instructions0ShotMalignancy % {'organ':organ,
                                                        'benign_tumors':benign_pancreas,
                                                        'malignant_tumors':malignant_pancreas,
                                                        'both_tumors':both_pancreas}
            elif organ=='kidney':
                instructions=instructions0ShotMalignancy % {'organ':organ,
                                                        'benign_tumors':benign_kidney,
                                                        'malignant_tumors':malignant_kidney,
                                                        'both_tumors':both_kidney}
        else:
            if organ=='liver':
                instructions=instructions0ShotMalignancyFast % {'organ':organ,
                                                        'benign_tumors':benign_liver,
                                                        'malignant_tumors':malignant_liver,
                                                        'both_tumors':both_liver}
            elif organ=='pancreas':
                instructions=instructions0ShotMalignancyFast % {'organ':organ,
                                                        'benign_tumors':benign_pancreas,
                                                        'malignant_tumors':malignant_pancreas,
                                                        'both_tumors':both_pancreas}
            elif organ=='kidney':
                instructions=instructions0ShotMalignancyFast % {'organ':organ,
                                                        'benign_tumors':benign_kidney,
                                                        'malignant_tumors':malignant_kidney,
                                                        'both_tumors':both_kidney}
    elif step=='malignant size':
        if len(examples)>0:
            raise ValueError('Examples not implemented for tumor size')
        if not fast:
            instructions=instructions0ShotMalignantSize % {'organ':organ,
                                                        'benign_tumors':benign_kidney,
                                                        'malignant_tumors':malignant_kidney,
                                                        'both_tumors':both_kidney,
                                                        'organ_locations':organ_part[organ]}
        else:
            raise ValueError('Fast not implemented for tumor size')
    elif step=='type and size':
        reports_and_answers={'kidney':[report_kidney,answer_kidney],
                             'liver':[report_liver,answer_liver],
                             'pancreas':[report_pancreas,answer_pancreas]}
        malignant_benign_both={'kidney':[benign_kidney,malignant_kidney,both_kidney],
                                 'liver':[benign_liver,malignant_liver,both_liver],
                                 'pancreas':[benign_pancreas,malignant_pancreas,both_pancreas]}
        instructions=instructions0ShotSizenType % {'organ':organ,
                                                    'benign_tumors':malignant_benign_both[organ][0],
                                                    'malignant_tumors':malignant_benign_both[organ][1],
                                                    'both_tumors':malignant_benign_both[organ][2],
                                                    'organ_locations':organ_part[organ],
                                                    'example_report':reports_and_answers[organ][0],
                                                    'example_answer':reports_and_answers[organ][1],
                                                    'extra_info':(summary_of_terms_pancreas if organ=='pancreas' else '')}
    elif step == 'HCC':
        malignant_benign_both={'liver':[benign_liver,malignant_liver,both_liver]}
        organ='liver'
        instructions=instructionsHCC % {'benign_tumors':malignant_benign_both[organ][0],
                                        'malignant_tumors':malignant_benign_both[organ][1],
                                        'both_tumors':malignant_benign_both[organ][2],
                                        'organ_locations':organ_part[organ],
                                        'example_report':reportHCC,
                                        'example_answer':answerHCC,
                                        'extra_info':''}
        

    elif step=='type and size pathology':
        reports_and_answers={'pancreas':[pathologyReportPancreas,pathologyReportPancreasAnswer]}
        malignant_benign_both={'pancreas':[benign_pancreas,malignant_pancreas,both_pancreas]}
        instructions=instructions0ShotSizenTypePathology % {'organ':organ,
                                                    'benign_tumors':malignant_benign_both[organ][0],
                                                    'malignant_tumors':malignant_benign_both[organ][1],
                                                    'both_tumors':malignant_benign_both[organ][2],
                                                    'organ_locations':organ_part[organ],
                                                    'example_report':reports_and_answers[organ][0],
                                                    'example_answer':reports_and_answers[organ][1],
                                                    'extra_info':(summary_of_terms_pancreas if organ=='pancreas' else '')}
    elif step=='type and size multi-organ':
        reports_and_answers={'kidney':[report_kidney,answer_kidney],
                             'liver':[report_liver,answer_liver],
                             'pancreas':[report_pancreas,answer_pancreas]}
        instructions=instructions0ShotSizenTypeMultiOrgan % {'organ':'pancreas',#used only as example
                                                             'example_report':report_pancreas,
                                                             'example_answer':answer_pancreas_multi_organ}
    elif step=='pathology':
        instructions=multi_tumor_pathology
    elif step=='pathology many reports':
        instructions=multi_tumor_pathology_multi
    elif step=='risk factors pancreas':
        instructions=risk_factors_pancreas
    elif step=='diagnoses':
        instructions=abnormality_prompt
    elif step=='refine normal pancreas':
        instructions = RefineNormalsKangPancreas
    elif step=='refine normal pancreas 2':
        instructions = RefineNormalsPancreasCancer
    elif step=='stage pancreas':
        instructions = pancreatic_tumor_staging_prompt
        
    return instructions

def create_conversation(data,target,target_data=None,examples=[],fast=True, 
                        step='tumor detection',organ='liver',row_name='Anon Report Text',
                        future_report=None):
    
    if target_data is None:
        target_data=data

    if step=='time machine':
        report,date=get_report_n_label(target_data,target,row_name=row_name,get_date=True)   
        print('Future report:',future_report)
        future_report,future_date=get_report_n_label(target_data,future_report,row_name=row_name,get_date=True)

        usr=time_machine_solver % {'organ':organ,
                                            'benign_tumors':benign_liver,
                                            'malignant_tumors':malignant_liver,
                                            'both_tumors':both_liver,
                                            'organ_locations':organ_part[organ],
                                            'date1':date,
                                            'date2':future_date,
                                            'report1':report,
                                            'report2':future_report}
        
    elif step=='longitudinal pancreas' or step=='longitudinal pancreas diagnosis':
        accessions,reports=get_longitudinal_reports(data,target)
        #concatenate the list of strings report
        reports='\n'.join(reports)
        if step=='longitudinal pancreas':
            usr = instructionsLongitudinalPancreas +'\n'+ reports
        else:
            usr = instructionsLongitudinalPancreasDiagnosis +'\n'+ reports
        message= [{"role": "system", "content": system+' \n '+observations},
                  {"role": "user", "content": usr}]
        return message,accessions
        
    elif step=='tumor slices':
        usr=get_instuctions(fast,step,examples=[],organ=organ,all_data=data,accession=target)
        message= [{"role": "system", "content": system+' \n '+observations},
                  {"role": "user", "content": usr}]
    else:
        usr=get_instuctions(fast,step,examples=examples,organ=organ)
        #add clinical notes
        usr+=' \n '
        usr+=observations
        usr+=' \n '
        #examples
        i=0
        print('Examples:',examples)
        for i,ex in enumerate(examples,1):
            report,label=get_report_n_label(data,ex,row_name=row_name)
            if report is None or label is None:
                raise ValueError('No label or report available for index '+str(ex))
            usr+='Report '+str(i)+': '+report+'\n '
            usr+='Report '+str(i)+' labels: '+label
            usr+=' \n --- \n '
        #target report
        i+=1
        if len(examples)==0:
            num=''
        else:
            num=' '+str(i)
        if step != 'find matching reports':
            report,_=get_report_n_label(target_data,target,row_name=row_name)
            usr+='Report'+num+': '+report+'\n '
            if report is None:
                raise ValueError('No report available for index '+str(target))
        else:
            report1,report2=get_report_n_label(target_data,target,row_name=row_name,get_2_reports=True,
                                               report2_col='radiology_findings_really')
            usr+='Report 1:\n'+report1+'\n '
            usr+='Report 2:\n'+report2+'\n '
            if (report1 is None) or (report2 is None):
                raise ValueError('No report available for index '+str(target))
        
        
            
        #question
        #usr+=question % {'last':num}

    message= [{"role": "system", "content": system+' \n '+observations},
                  {"role": "user", "content": usr}]

    #print('Report:',report)
    
    return message

def multi_prompt_message(data,target,target_data,
                         per_message_examples=5,examples=[]):
    assert ((len(examples)+1)%(per_message_examples+1))==0
    num_examples=per_message_examples
    step=per_message_examples+1
    
    message= [
        {"role": "system", "content": system+' \n '+observations},
        #{"role": "user", "content": usr},
    ]
    #print(int(len(examples)/step))
    for i in range(int(len(examples)/step)):
        ex=examples[i*step:(i+1)*step-1]
        t=examples[(i+1)*step-1]
        m=create_conversation(data=data,target=t,examples=ex)[1]["content"]
        message.append({"role": "user", "content": m})
        _,l=get_report_n_label(data,t)
        message.append({"role": "assistant", "content": l})
        
    ex=examples[-per_message_examples:]
    m=create_conversation(data=data,target_data=target_data,
                          target=target,examples=ex)[1]["content"]
    message.append({"role": "user", "content": m})
    
    return message

    
        
    
def run_model(message,base_url='http://0.0.0.0:8000/v1',labels=None,id=None,batch=1):
    conver,answer=SendMessageAPI(text=None,conver=message,base_url=base_url,labels=labels,id=id,batch=batch)
    return answer

def run(target,examples,data,target_data=None,base_url='http://0.0.0.0:8000/v1',print_message=False,
        step='tumor detection',organ='liver',fast=True,row_name='Anon Report Text',id_column='Anon Acc #',
        future_report=None):
    if target_data is None:
        target_data=data

    if isinstance(target,list):
        message=[]
        labels=[]
        id=[]
        for tgt in target:
            message.append(create_conversation(data=data,target=tgt,examples=examples, target_data=target_data,step=step,organ=organ,fast=fast,
                                               row_name=row_name,future_report=future_report))
            if 'Pancreas Tumor' not in data.columns:
                labels=None
            else:
                labels.append(data.iloc[target][['Liver Tumor','Kidney Tumor','Pancreas Tumor']])
            id.append(data.iloc[target][id_column])
        batch=len(message)
        print('Batch:',batch)
    else:
        message=create_conversation(data=data,target=target,examples=examples,
                                    target_data=target_data,step=step,organ=organ,fast=fast,
                                               row_name=row_name,future_report=future_report)
        if step == 'longitudinal pancreas' or step == 'longitudinal pancreas diagnosis':
            message,accessions=message
        print()
        print('Message:',message)
        print()
        #check if the columns are present
        if 'Pancreas Tumor' not in data.columns:
            labels=None
        else:
            labels=data.iloc[target][['Liver Tumor','Kidney Tumor','Pancreas Tumor']]
        #print('ID column:',id_column)
        try:
            id=data.iloc[target][id_column]
        except:
            id=None
        batch=1
    if print_message:
        print(message)

    if step == 'longitudinal pancreas' or step == 'longitudinal pancreas diagnosis':
        return run_model(message,base_url=base_url,labels=None,id=None,batch=batch),accessions
    else:
        return run_model(message,base_url=base_url,labels=labels,id=id,batch=batch)


def run_multi_prompt(target,examples,data,target_data=None,
                     base_url='http://0.0.0.0:8000/v1',print_message=False,
                     per_message_examples=5):
    if target_data is None:
        target_data=data
    message=multi_prompt_message(data=data,target=target,examples=examples,
                                 per_message_examples=per_message_examples,
                                 target_data=target_data)
    if print_message:
        print(message)
    return run_model(message,base_url=base_url)

def get_value_old(pattern, string):
    match = re.search(pattern, string)
    if match:
        return int(match.group(1))
    else:
        return np.nan
    

def interpret_output_old(string):
    liver_pattern = r'liver tumor=(\d+)'
    kidney_pattern = r'kidney tumor=(\d+)'
    pancreas_pattern = r'pancreas tumor=(\d+)'

    return {'Liver Tumor':get_value(liver_pattern,string),
            'Kidney Tumor':get_value(kidney_pattern,string),
            'Pancreas Tumor':get_value(pancreas_pattern,string)}

def get_value(pattern, string,step='tumor detection'):
    matches = re.findall(pattern, string.lower())
    print('Matches:',matches)

    if len(matches)==0:
        return np.nan

    if step == 'malignant size' or step == 'all sizes':
        sizes = []

        for match in matches:
            # Extract both integers and floating point numbers
            match,unit=match
            print('Match:',match)
            print('Unit:',unit)
            numbers = [float(num) for num in re.findall(r'\d+\.\d+|\d+', match)]
            print('Numbers:',numbers)
            
            if len(numbers) == 0:
                print('No numbers found')
                continue

            # Convert to mm depending on whether 'cm' or 'mm' is present
            for num in numbers:
                if unit=='cm':
                    sizes.append(num * 10)  # Convert cm to mm
                elif unit=='mm':
                    sizes.append(num)  # Already in mm
            print('Sizes:',sizes)

        # Return the largest size in mm, or np.nan if no sizes were found
        if len(sizes) == 0:
            return np.nan
        else:
            if step == 'malignant size':
                return np.max(sizes)
            else:
                return sizes
                #sz=''
                #for s in sizes:
                #    sz+=str(s)+' '
                #return str(sizes).replace('[','').replace(']','')

    else:
        if 'yes' in matches[0]:
            return 1
        elif 'no' in matches[0]:
            return 0
        else:
            return np.nan
        
def extract_liver_tumors(answer_text, organ="liver"):
    """
    1) Splits the answer by lines that contain "liver tumor N:" to isolate each chunk.
    2) For each chunk, parses the mandatory fields: type, certainty, size, location,
       arterial enhancement, washout, capsule, threshold growth, LI-RADS.
    3) Raises an error if any field is missing.
    4) Returns a dict like:
         {
           "liver tumor 1": {
              "type": ...,
              "certainty": ...,
              "size": ...,
              ...
           },
           "liver tumor 2": {...},
           ...
         }
    """

    # Regex to split on lines like "liver tumor 1:", capturing the line as a separate token.
    # Use re.IGNORECASE so it catches "Liver Tumor 1:" or "LIVER TUMOR 1:", etc.
    split_pattern = rf"(?i)(?=(?:{organ}\s+tumor\s+\d+:))"

    # Split the answer into chunks, each starting with "liver tumor N:"
    # The first chunk could be empty if the text starts with "liver tumor 1:" pattern. 
    chunks = re.split(split_pattern, answer_text)
    # Filter out empty chunks or whitespace.
    chunks = [c.strip() for c in chunks if c.strip()]

    tumors_dict = {}
    for chunk in chunks:
        # The first line in chunk should look like "liver tumor 1:"
        # Extract the tumor number from that line.
        # We'll do a quick check:
        first_line_match = re.match(rf"(?i){organ}\s+tumor\s+(\d+):", chunk)
        if not first_line_match:
            # This chunk doesn't start with "liver tumor N:", skip or continue
            continue

        tumor_number = first_line_match.group(1).strip()
        tumor_key = f"{organ} tumor {tumor_number}"

        # For each field, we look for e.g. "type = X;"
        # We use a small helper function.
        def find_field(field_name):
            # pattern: e.g. "type = ???;"
            # we allow optional whitespace.  We capture up to the next semicolon
            field_pattern = rf"{field_name}\s*=\s*([^;]+);"
            m = re.search(field_pattern, chunk, re.IGNORECASE)
            if not m:
                raise ValueError(f"Missing field '{field_name}' in chunk:\n{chunk}")
            return m.group(1).strip()

        # Now we get each mandatory field
        type_raw        = find_field("type")
        certainty_raw   = find_field("certainty")
        size_raw        = find_field("size")
        location_raw    = find_field("location")
        arterial_raw    = find_field("arterial enhancement")
        washout_raw     = find_field("washout")
        capsule_raw     = find_field("capsule")
        threshold_raw   = find_field("threshold growth")
        lirads_raw      = find_field("LI-RADS")

        # parse size (like your earlier code)
        if 'multiple' in size_raw.lower():
            size_parsed = 'multiple'
        else:
            size_parsed = get_value(r"(.*?)(cm|mm)", size_raw, step='malignant size')

        tumors_dict[tumor_key] = {
            "type"               : type_raw,
            "certainty"          : certainty_raw,
            "size"               : size_parsed,
            "location"           : location_raw,
            "arterial enhancement": arterial_raw,
            "washout"            : washout_raw,
            "capsule"            : capsule_raw,
            "threshold growth"   : threshold_raw,
            "LI-RADS"            : lirads_raw,
        }

    return tumors_dict

def parse_and_normalize_risk(text):
    """
    Extract, normalize, and type-coerce the risk JSON from an LLM reply.
    Categorical fields map to {'yes','no','absent'}; numbers become floats or None;
    lists normalized; BMI auto-computed if height+weight are present.

    Backward compatibility:
    - If present, 'epigastric_back_pain_anorexia_steatorrhea' is propagated to
      ('epigastric_pain','back_pain','anorexia','steatorrhea') when those are missing.
    - If present, 'painless_cholestasis_symptoms' is propagated to
      ('painless_jaundice','pruritus','dark_urine','pale_acholic_stools') when missing.
    """

    import json, re

    # Canonical keys (keep in sync with the prompt)
    CANONICAL_KEYS = [
        "smoking", "obesity", "patient_weight_kg", "patient_height_cm", "bmi",
        "alcohol_use", "chronic_pancreatitis", "acute_pancreatitis", "pancreatitis",
        "family_history_pancreatic_cancer",
        "high_risk_germline_mutation", "high_risk_mutation_list",
        "diabetes", "weight_loss",
        "painless_jaundice", "pruritus", "dark_urine", "pale_acholic_stools",
        "epigastric_pain", "back_pain", "anorexia", "steatorrhea",
        "dvt_or_pe", "abdominal_pain",
        "high_direct_bilirubin_or_alp", "direct_bilirubin_value", "direct_bilirubin_unit",
        "alp_value", "alp_unit",
        "elevated_ca19_9", "ca19_9_value", "ca19_9_unit",
        "high_nlr_or_thrombocytosis_or_low_albumin",
        "nlr", "anc_value", "anc_unit", "alc_value", "alc_unit",
        "platelets_value", "platelets_unit",
        "albumin_value", "albumin_unit",
        "pancreatic_duct_dilatation", "pancreatic_focal_atrophy_or_contour_change",
        "ipmn", "pancreatic_cyst",
    ]

    CATEGORICAL_KEYS = {
        "smoking","obesity","alcohol_use",
        "chronic_pancreatitis","acute_pancreatitis","pancreatitis",
        "family_history_pancreatic_cancer",
        "high_risk_germline_mutation",
        "diabetes","weight_loss",
        "painless_jaundice","pruritus","dark_urine","pale_acholic_stools",
        "epigastric_pain","back_pain","anorexia","steatorrhea",
        "dvt_or_pe","abdominal_pain",
        "high_direct_bilirubin_or_alp",
        "elevated_ca19_9",
        "high_nlr_or_thrombocytosis_or_low_albumin",
        "pancreatic_duct_dilatation","pancreatic_focal_atrophy_or_contour_change",
        "ipmn","pancreatic_cyst",
    }

    NUMERIC_KEYS = {
        "patient_weight_kg","patient_height_cm","bmi",
        "direct_bilirubin_value","alp_value",
        "ca19_9_value",
        "nlr","anc_value","alc_value","platelets_value","albumin_value",
    }
    LIST_KEYS = {"high_risk_mutation_list"}

    # Aliases / legacy keys
    KEY_ALIASES = {
        "main_pancreatic_duct_dilatation": "pancreatic_duct_dilatation",
        "any_pancreatitis": "pancreatitis",
        "new_onset_diabetes_ge_50": "diabetes",
        # composites handled below
    }

    # ----- helpers -----
    def _ensure_text(x) -> str:
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            if "content" in x and isinstance(x["content"], str):
                return x["content"]
            if "message" in x and isinstance(x["message"], dict) and "content" in x["message"]:
                return x["message"]["content"]
            if "choices" in x and x["choices"]:
                c = x["choices"][0]
                if isinstance(c, dict):
                    return c.get("text") or c.get("message", {}).get("content", "") or str(x)
            return str(x)
        if isinstance(x, (list, tuple)):
            return "\n".join(_ensure_text(e) for e in x)
        return str(x)

    def _strip_code_fences(s: str) -> str:
        s = s.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        return s

    def _extract_json_block(raw: str) -> str:
        m = re.search(r"<BEGIN_JSON>(.*?)<END_JSON>", raw, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return _strip_code_fences(m.group(1))
        seg = _strip_code_fences(raw)
        start = seg.find("{")
        if start == -1:
            return ""
        depth = 0
        for i, ch in enumerate(seg[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return seg[start:i+1].strip()
        return ""

    def _safe_json_loads(s: str):
        try:
            return json.loads(s)
        except Exception:
            pass
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            pass
        s2 = re.sub(r",\s*([}\]])", r"\1", s)
        try:
            return json.loads(s2)
        except Exception:
            return {}

    def _normalize_keys(d: dict) -> dict:
        out = {}
        for k, v in (d or {}).items():
            out[KEY_ALIASES.get(k, k)] = v
        return out

    def _norm_cat_value(v):
        if v is None: return "absent"
        s = (v if isinstance(v, str) else str(v)).strip().lower()
        if s in {"yes","no","absent"}: return s
        if s in {"y","true","present","positive"}: return "yes"
        if s in {"n","false","negative"}: return "no"
        if s in {"", "u", "unk", "unknown", "na", "n/a", "none", "not mentioned", "no mention"}:
            return "absent"
        return "absent"

    def _coerce_numbers_and_lists(d: dict) -> dict:
        dd = dict(d)
        for k in NUMERIC_KEYS:
            if k in dd:
                try:
                    dd[k] = float(dd[k]) if dd[k] is not None else None
                except Exception:
                    dd[k] = None
        for k in LIST_KEYS:
            if k in dd:
                if dd[k] is None:
                    dd[k] = []
                elif not isinstance(dd[k], list):
                    dd[k] = [dd[k]]
            else:
                dd[k] = []
        return dd

    # ----- main -----
    text = _ensure_text(text)
    if "</think>" in text:
        text = text.split("</think>")[-1]

    raw = _extract_json_block(text)
    parsed = _safe_json_loads(raw) if raw else {}

    parsed = _normalize_keys(parsed)
    parsed = _coerce_numbers_and_lists(parsed)

    # Back-compat: propagate composite fields if present
    combo1 = "epigastric_back_pain_anorexia_steatorrhea"
    if combo1 in parsed:
        val = _norm_cat_value(parsed.get(combo1))
        for k in ("epigastric_pain","back_pain","anorexia","steatorrhea"):
            if k not in parsed or parsed[k] is None:
                parsed[k] = val

    combo2 = "painless_cholestasis_symptoms"
    if combo2 in parsed:
        val = _norm_cat_value(parsed.get(combo2))
        for k in ("painless_jaundice","pruritus","dark_urine","pale_acholic_stools"):
            if k not in parsed or parsed[k] is None:
                parsed[k] = val

    # Normalize categoricals to yes/no/absent
    for k in CATEGORICAL_KEYS:
        parsed[k] = _norm_cat_value(parsed.get(k))

    # Derive 'pancreatitis' from components when useful
    if parsed.get("pancreatitis", "absent") in {"absent", None}:
        ch = parsed.get("chronic_pancreatitis", "absent")
        ac = parsed.get("acute_pancreatitis", "absent")
        if ch == "yes" or ac == "yes":
            parsed["pancreatitis"] = "yes"
        elif ch == "no" and ac == "no":
            parsed["pancreatitis"] = "no"

    # Auto-compute BMI if missing but feasible
    if parsed.get("bmi") is None:
        wt = parsed.get("patient_weight_kg")
        ht_cm = parsed.get("patient_height_cm")
        if isinstance(wt, (int, float)) and isinstance(ht_cm, (int, float)) and wt > 0 and ht_cm > 0:
            m = ht_cm / 100.0
            try:
                parsed["bmi"] = round(wt / (m*m), 1)
            except Exception:
                parsed["bmi"] = None

    # Ensure all canonical keys exist (categoricals default to 'absent')
    clean = {k: parsed.get(k, ("absent" if k in CATEGORICAL_KEYS else None)) for k in CANONICAL_KEYS}
    return clean

import json, re

def parse_pathology_json(text):
    """
    Robust parser for the STRICT-JSON pathology output.
    Unknowns are 'u'. Arrays/dicts are accepted even if double-encoded as JSON strings.
    """

    ORG_ENUM  = {"liver","kidney","pancreas","bladder","gallbladder","esophagus","stomach","duodenum","colon","prostate","uterus","spleen","adrenal gland","other"}
    SIDE_ENUM = {"left","right","bilateral","unspecified"}

    # ----------------- helpers (same style as your other parser) -----------------
    def _ensure_text(x) -> str:
        if isinstance(x, str): return x
        if isinstance(x, dict):
            if "content" in x and isinstance(x["content"], str): return x["content"]
            if "message" in x and isinstance(x["message"], dict) and "content" in x["message"]: return x["message"]["content"]
            if "choices" in x and x["choices"]:
                c = x["choices"][0]
                if isinstance(c, dict): return c.get("text") or c.get("message", {}).get("content", "") or str(x)
            return str(x)
        if isinstance(x, (list, tuple)): return "\n".join(_ensure_text(e) for e in x)
        return str(x)

    def _strip_code_fences(s: str) -> str:
        s = s.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
        s = re.sub(r"\s*```$", "", s)
        return s

    def _extract_json_block(raw: str) -> str:
        m = re.search(r"<BEGIN_JSON>(.*?)<END_JSON>", raw, flags=re.S | re.I)
        if m: return _strip_code_fences(m.group(1))
        seg = _strip_code_fences(raw)
        start = seg.find("{")
        if start == -1: return ""
        depth = 0
        for i, ch in enumerate(seg[start:], start=start):
            if ch == "{": depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0: return seg[start:i+1].strip()
        return ""

    def _safe_json_loads(s: str):
        for candidate in (s, s.replace("'", '"'), re.sub(r",\s*([}\]])", r"\1", s)):
            try:
                return json.loads(candidate)
            except Exception:
                pass
        return {}

    def _norm_tri(v) -> str:
        if isinstance(v, bool): return 'yes' if v else 'no'
        s = (v if isinstance(v, str) else str(v)).strip().lower()
        if s in {'true','yes','present','positive','1'}: return 'yes'
        if s in {'false','no','negative','0'}: return 'no'
        if s in {'u','unk','unknown','na','n/a','none','indeterminate','undecidable','not sure','not specified',''}: return 'u'
        return 'u'

    def _norm_organ(x: str) -> str:
        s = (x or '').strip().lower()
        if s in {'adrenal','adrenal glands','adrenal gland'}: s = 'adrenal gland'
        return s if s in ORG_ENUM else 'other'

    def _norm_side(x: str) -> str:
        s = (x or '').strip().lower()
        return s if s in SIDE_ENUM else 'unspecified'

    def _as_list(x):
        if x is None: return []
        return x if isinstance(x, list) else [x]

    def _citations_list(x):
        return [str(c).strip() for c in _as_list(x) if str(c).strip()]

    def _maybe_json_coerce(x):
        """If x is a JSON-looking string for a list/dict, parse it; else return x."""
        if isinstance(x, str):
            sx = x.strip()
            if sx and sx[0] in "[{" and sx[-1] in "]}":
                try:
                    return json.loads(sx)
                except Exception:
                    return x
        return x

    # ----------------- main parse -----------------
    text = _ensure_text(text)
    if "</think>" in text:
        text = text.split("</think>")[-1]

    raw = _extract_json_block(text)
    data = _safe_json_loads(raw) if raw else {}

    # Patient-level malignancy decision — accept dict OR bare bool/string
    p = data.get("patient_has_malignant_tumor")
    if isinstance(p, dict):
        value_raw = p.get("value")
        decision_rationale = (p.get("rationale") or "").strip()
        decision_citations = _citations_list(p.get("citations"))
    else:
        value_raw = p  # could be bool or string
        decision_rationale = ""
        decision_citations = []
    decision = _norm_tri(value_raw)

    out = {
        "patient_has_malignant_tumor": decision,                        # 'yes'|'no'|'u'
        "patient_has_malignant_tumor_rationale": decision_rationale,    # short text
        "patient_has_malignant_tumor_citations": decision_citations,    # list[str]
        "malignant_organs": [],
        "metastasis_organs": [],
        "benign_mass_or_cyst": [],
        "tumor_types": []
    }

    # malignant_organs (allow stringified JSON)
    mo = _maybe_json_coerce(data.get("malignant_organs"))
    for item in _as_list(mo):
        if isinstance(item, str):  # tolerate sloppy string element
            item = {"organ": item, "side": "unspecified", "rationale": "", "citations": []}
        out["malignant_organs"].append({
            "organ": _norm_organ(item.get("organ")),
            "side": _norm_side(item.get("side")),
            "rationale": (item.get("rationale") or "").strip(),
            "citations": _citations_list(item.get("citations")),
        })

    # metastasis_organs
    me = _maybe_json_coerce(data.get("metastasis_organs"))
    for item in _as_list(me):
        if isinstance(item, str):
            item = {"organ": item, "rationale": "", "citations": []}
        out["metastasis_organs"].append({
            "organ": _norm_organ(item.get("organ")),
            "rationale": (item.get("rationale") or "").strip(),
            "citations": _citations_list(item.get("citations")),
        })

    # benign_mass_or_cyst
    bm = _maybe_json_coerce(data.get("benign_mass_or_cyst"))
    for item in _as_list(bm):
        if isinstance(item, str):
            item = {"organ": "other", "finding": item, "rationale": "", "citations": []}
        out["benign_mass_or_cyst"].append({
            "organ": _norm_organ(item.get("organ")),
            "finding": (item.get("finding") or "").strip(),
            "rationale": (item.get("rationale") or "").strip(),
            "citations": _citations_list(item.get("citations")),
        })

    # tumor_types
    tt = _maybe_json_coerce(data.get("tumor_types"))
    seen = set()
    for t in _as_list(tt):
        if not isinstance(t, str): continue
        t_clean = t.strip()
        if not t_clean: continue
        key = t_clean.lower()
        if key not in seen:
            seen.add(key)
            out["tumor_types"].append(t_clean)

    # If JSON missing entirely, keep a safe default
    if not raw and out["patient_has_malignant_tumor"] == 'u':
        # nothing else to do; out stays minimal
        pass

    return out


# --- helper: normalize & extract bracketed organ lists from "many reports" outputs ---
def _normalize_organ_name(s: str) -> str:
    s0 = (s or "").strip().strip(" ,;.:\"'()").lower()
    s0 = re.sub(r"\s+", " ", s0)

    # quick synonym map (extend as needed)
    syn = {
        "oesophagus": "esophagus",
        "esophageal": "esophagus",
        "gastric": "stomach",
        "hepatic": "liver",
        "gb": "gallbladder",
        "gall bladder": "gallbladder",
        "endometrium": "uterus",
        "endometrial": "uterus",
        "splenic": "spleen",
        "duodenal": "duodenum",
        "colonic": "colon",
        "pancreatic": "pancreas",
        "prostatic": "prostate",
        "right adrenal": "right adrenal gland",
        "left adrenal": "left adrenal gland",
        "adrenal": "adrenal gland",
        "adrenals": "adrenal gland",
        "left renal": "left kidney",
        "right renal": "right kidney",
        "renal (left)": "left kidney",
        "renal (right)": "right kidney",
    }
    if s0 in syn:
        return syn[s0]

    # normalize patterns like "left adrenal gland", "right kidney" already OK
    return s0

def _extract_organ_list_for_key(text: str, key: str):
    """
    Look for lines like:
      organs_with_primary_malignant_tumors: [liver, left kidney, bladder]
    Works across line breaks inside the brackets. Returns a de-duplicated, normalized list.
    """
    # Try bracketed capture first (multi-line, non-greedy).
    m = re.search(rf"(?im)^\s*{re.escape(key)}\s*:\s*\[(.*?)\]", text, flags=re.S)
    items = []
    if m:
        inner = m.group(1)
        # Try JSON first (e.g., ["liver","bladder"])
        try:
            maybe = json.loads("[" + inner + "]")
            if isinstance(maybe, list):
                items = [str(x) for x in maybe]
        except Exception:
            # Fallback: split on commas/semicolons
            parts = re.split(r"[;,]", inner)
            items = [p.strip() for p in parts if p.strip()]
    else:
        # Accept explicit "none"/"na"/"unknown" etc. as empty
        if re.search(rf"(?im)^\s*{re.escape(key)}\s*:\s*(none|no\s+organs|na|n/?a|u|unknown)\b", text):
            items = []
        else:
            # No match found; return empty list
            items = []

    # normalize + dedupe while preserving order
    seen = set()
    out = []
    for it in items:
        norm = _normalize_organ_name(it)
        if norm and norm not in seen:
            seen.add(norm)
            out.append(norm)
    return out
import json, re

def parse_pancreas_stage_json(text):
    """
    Extract the STRICT JSON for pancreatic tumor staging from an LLM reply.

    Expected top-level keys in the JSON:
      - no_pancreatic_tumor: bool
      - no_pancreatic_tumor_justification: str
      - no_pancreatic_tumor_supporting_quotes: list[str]
      - tumors: list[dict]

    Post-processing:
      - Convert each tumor['size_mm'] to a string like "46 x 33"
        instead of a list [46, 33] or a list-looking string "[46, 33]".
    """

    # ---------- small helpers ----------
    def _ensure_text(x) -> str:
        if isinstance(x, str):
            return x
        if isinstance(x, dict):
            if "content" in x and isinstance(x["content"], str):
                return x["content"]
            if "message" in x and isinstance(x["message"], dict) and "content" in x["message"]:
                return x["message"]["content"]
            if "choices" in x and x["choices"]:
                c = x["choices"][0]
                if isinstance(c, dict):
                    return c.get("text") or c.get("message", {}).get("content", "") or str(x)
            return str(x)
        if isinstance(x, (list, tuple)):
            return "\n".join(_ensure_text(e) for e in x)
        return str(x)

    def _strip_code_fences(s: str) -> str:
        s = s.strip()
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```$", "", s)
        return s

    def _extract_json_block(raw: str) -> str:
        # Prefer explicitly marked <BEGIN_JSON> ... <END_JSON> if present
        m = re.search(r"<BEGIN_JSON>(.*?)<END_JSON>", raw, flags=re.DOTALL | re.IGNORECASE)
        if m:
            return _strip_code_fences(m.group(1))

        # Otherwise, strip code fences, then try to grab the first balanced {...}
        seg = _strip_code_fences(raw)
        start = seg.find("{")
        if start == -1:
            return ""
        depth = 0
        for i, ch in enumerate(seg[start:], start=start):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return seg[start:i+1].strip()
        return ""

    def _safe_json_loads(s: str):
        if not s:
            return {}
        # Try as-is
        try:
            return json.loads(s)
        except Exception:
            pass
        # Try with single quotes replaced
        try:
            return json.loads(s.replace("'", '"'))
        except Exception:
            pass
        # Try removing trailing commas
        s2 = re.sub(r",\s*([}\]])", r"\1", s)
        try:
            return json.loads(s2)
        except Exception:
            return {}

    def _fmt_num(n):
        try:
            f = float(n)
            if f.is_integer():
                return str(int(f))
            else:
                return str(f)
        except Exception:
            return str(n)

    def _format_size_list_to_str(size_val):
        """
        Convert:
          [46, 33]     -> "46 x 33"
          "[46, 33]"   -> "46 x 33"
          "[10]"       -> "10"
        If already a simple string (e.g. "10 x 12"), return as is.
        If empty or invalid, return "".
        """
        # case 1: size is a string
        if isinstance(size_val, str):
            s = size_val.strip()
            # string that looks like a JSON list, e.g. "[10]" or "[10, 12]"
            if s.startswith("[") and s.endswith("]"):
                try:
                    parsed = json.loads(s)
                    if isinstance(parsed, (list, tuple)) and parsed:
                        return " x ".join(_fmt_num(x) for x in parsed)
                except Exception:
                    # if parsing fails, just fall back to the raw string without brackets
                    inner = s.strip("[]").strip()
                    if not inner:
                        return ""
                    # split on comma if present
                    parts = [p.strip() for p in inner.split(",") if p.strip()]
                    if len(parts) == 0:
                        return ""
                    return " x ".join(parts)
            # plain string that doesn't look like a list — keep it
            return s

        # case 2: list/tuple
        if isinstance(size_val, (list, tuple)) and len(size_val) > 0:
            return " x ".join(_fmt_num(x) for x in size_val)

        # anything else → no size
        return ""

    # ---------- main ----------
    text = _ensure_text(text)

    # Strip chain-of-thought if model leaks it
    if "</think>" in text:
        text = text.split("</think>")[-1]

    raw_json_str = _extract_json_block(text)
    data = _safe_json_loads(raw_json_str)

    if not isinstance(data, dict):
        data = {}

    # Safe default structure
    out = {
        "no_pancreatic_tumor": bool(data.get("no_pancreatic_tumor", False)),
        "no_pancreatic_tumor_justification": data.get("no_pancreatic_tumor_justification", "") or "",
        "no_pancreatic_tumor_supporting_quotes": data.get("no_pancreatic_tumor_supporting_quotes", []) or [],
        "tumors": data.get("tumors", []) or [],
    }

    # Ensure types for the top-level lists
    if not isinstance(out["no_pancreatic_tumor_supporting_quotes"], list):
        out["no_pancreatic_tumor_supporting_quotes"] = [str(out["no_pancreatic_tumor_supporting_quotes"])]
    if not isinstance(out["tumors"], list):
        out["tumors"] = []

    # --- convert size_mm -> "a x b x c" clean string ---
    for t in out["tumors"]:
        if isinstance(t, dict) and "size_mm" in t:
            t["size_mm"] = _format_size_list_to_str(t.get("size_mm"))

    return out

def interpret_output(string,step='tumor detection',organ='liver'):
    # -- strip LLM chain-of-thought locally, NOT in dataframe
    if "</think>" in string:
        string = string.split("</think>")[-1]

    if step=='tumor detection':
        liver_pattern = r'liver tumor presence\s*[=:]\s*.*?(?:;|$|,|/|yes|no|u)'
        kidney_pattern = r'kidney tumor presence\s*[=:]\s*.*?(?:;|$|,|/|yes|no|u)'
        pancreas_pattern = r'pancreas tumor presence\s*[=:]\s*.*?(?:;|$|,|/|yes|no|u)'

        return {'Liver Tumor':get_value(liver_pattern,string),
                'Kidney Tumor':get_value(kidney_pattern,string),
                'Pancreas Tumor':get_value(pancreas_pattern,string)}
    elif step == 'tumor slices':
        # --- keep only the filled list ---
        filled = string.split("**JUSTIFICATION**", 1)[0]

        pattern = re.compile(
            r'tumor\s*id\s*=\s*([^;]+?)\s*;'      # full ID up to semicolon
            r'.*?series\s*=\s*([^\s;,]+)'         # series number
            r'.*?image\s*=\s*([^\s;,]+)',         # image number
            flags=re.I | re.S
        )

        triples = pattern.findall(filled)
        if not triples:
            raise ValueError(f"No tumour lines detected — check LLM output format:\n{string}")

        def _num_or_nan(x: str):
            x = x.strip().rstrip(',;')
            if re.fullmatch(r'u|unknown|na|nan', x, flags=re.I):
                return np.nan
            try:
                return int(x)
            except ValueError:
                return np.nan

        records = [
            {
                "Tumor ID": id_.strip().lower(),   # normalise case if you wish
                "Series"  : _num_or_nan(se),
                "Image"   : _num_or_nan(im),
            }
            for id_, se, im in triples
        ]

        return pd.DataFrame(records)
    elif step=='pre-diagnostic confirmation':
        # Define regex patterns to capture yes/no answers.
        tumor_pattern = r'pancreatic tumor suspicion\s*[=:]\s*?(?:;|$|,|/|yes|no)'
        surgery_pattern = r'pancreas surgery\s*[=:]\s*?(?:;|$|,|/|yes|no)'
        cancer_pattern = r'cancer history\s*[=:]\s*?(?:;|$|,|/|yes|no)'
        
        return {'Pancreatic Tumor Suspicion': get_value(tumor_pattern, string),
                'Pancreas Surgery': get_value(surgery_pattern, string),
                'Cancer History': get_value(cancer_pattern, string)}
    elif step=='find matching reports':
        match_pattern = r'same report\s*[=:]\s*?(?:;|$|,|/|yes|no)'
        return {'Matching Reports': get_value(match_pattern, string)}
    elif step=='malignancy detection':
        pattern = r"malignant tumor in %s\s*[=:]\s*.*?(?:;|$|,|/|yes|no|u)" % organ  
        return {'Malignant Tumor in '+organ:get_value(pattern,string)}
    elif step=='malignant size':
        pattern = r"%s malignant tumor size\s*[=:]\s*(.*?)(cm|mm)" % organ
        y={'Malignant Tumor in '+organ:get_value(pattern,string,step=step)}
        print(y)
        return y
    elif step=='time machine':
        malignancy_pattern = r"very likely malignancy in %s in the first exam\s*[=:]\s*.*?(?:;|$|,|/|yes|no|u)" % organ
        size_pattern = r"%s malignant tumor size\s*[=:]\s*(.*?)(cm|mm)" % organ
        return {'very likely malignancy in '+organ:get_value(malignancy_pattern,string),
                'very likely malignant tumor in '+organ:get_value(size_pattern,string,step='malignant size')}
    elif step == 'type and size' or step=='type and size pathology':
        # Extracting multiple tumors from the LLM output
        tumor_pattern = rf"{organ} tumor \d+: type = (?P<type>.+?); certainty = (?P<certainty>.+?); size = (?P<size>.+?); location = (?P<location>.+?);"
        matches = re.finditer(tumor_pattern, string.lower())
        
        tumors = {}
        for match in matches:
            tumor_key = f"{organ} tumor {len(tumors) + 1}"
            size_raw = match.group('size').strip()
            if 'multiple' in size_raw:
                size_numbers = 'multiple'
            else:
                size_numbers = get_value(r"(.*?)(cm|mm)", size_raw, step='malignant size')
            tumors[tumor_key] = {
                'type': match.group('type').strip(),
                'certainty': match.group('certainty').strip(),
                'size': size_numbers,
                'location': match.group('location').strip(),
            }
        return tumors
    elif step == 'pathology':
        return parse_pathology_json(string)
    
    elif step == 'pathology many reports':
        """
        Expects LLM output that includes four lines (order-insensitive), e.g.:

        organs_with_primary_malignant_tumors: [liver, left kidney, bladder]
        organs_with_metastatic_tumors: [bladder, esophagus]
        organs_with_benign_tumors: [liver, prostate]
        organs_with_unknown_tumor_type: [spleen]

        Followed by a 'Justification:' section (ignored here).
        """
        
        keys = [
            "organs_with_primary_malignant_tumors",
            "organs_with_metastatic_tumors",
            "organs_with_benign_tumors",
            "organs_with_unknown_tumor_type",
        ]

        results = {}
        for k in keys:
            results[k] = _extract_organ_list_for_key(string, k)


        # ensure all four canonical outputs are present
        for required in [
            "organs_with_primary_malignant_tumors",
            "organs_with_metastatic_tumors",
            "organs_with_benign_tumors",
            "organs_with_unknown_tumor_type",
        ]:
            results.setdefault(required, [])

        return results
    
    elif step == 'HCC':
        out = extract_liver_tumors(string.lower(),organ='liver')
        #print(out)
        #raise ValueError('Stop here')
        return out
    elif step == 'risk factors pancreas':
        return parse_and_normalize_risk(string)
    elif step == 'type and size multi-organ':
        # Extracting multiple tumors from the LLM output
        if ("No lesions mentioned." in string)  and ('lesion 1:' not in string.lower()):
            return {'no lesion': {
                'type': 'no lesion',
                'certainty':  'no lesion',
                'size':  'no lesion',
                'location':  'no lesion',
                'organ': 'no lesion',
                'attenuation':  'no lesion',
            }}
        tumor_pattern = rf"lesion \d+: type = (?P<type>.+?); certainty = (?P<certainty>.+?); size = (?P<size>.+?); organ = (?P<organ>.+?); location = (?P<location>.+?); attenuation = (?P<attenuation>.+?);"
        matches = re.finditer(tumor_pattern, string.lower())
        
        tumors = {}
        for match in matches:
            tumor_key = f"tumor {len(tumors) + 1}"
            size_raw = match.group('size').strip()
            if 'multiple' in size_raw:
                size_numbers = 'multiple'
            elif 'tiny' in size_raw:
                size_numbers = 'tiny'
            elif 'massive' in size_raw:
                size_numbers = 'massive'
            elif size_raw.lower() in ['u', ' u', ' u', 'unk', 'unkn', 'unknown', 'n/a', 'na', 'not available']:
                size_numbers = 'u'
            else:
                size_numbers = get_value(r"(.*?)(cm|mm)", size_raw, step='all sizes')
            tumors[tumor_key] = {
                'type': match.group('type').strip(),
                'certainty': match.group('certainty').strip(),
                'size': size_numbers,
                'location': match.group('location').strip(),
                'organ': match.group('organ').strip(),
                'attenuation': match.group('attenuation').strip()
            }
        return tumors
    elif step == 'diagnoses':
        if "abnormalities =" in string:
            start_index = string.rfind("abnormalities =") + len("abnormalities =")
        elif "abnormalities=" in string:
            start_index = string.rfind("abnormalities=") + len("abnormalities=")
        elif "[" in string:
            start_index = string.find("[")
        else:
            return None
        end_index = string.rfind("]", start_index) + 1  # Include the closing bracket
        abnormalities_str = string[start_index:end_index].strip()
        
        # Safely evaluate the string as a Python object
        #abnormalities = ast.literal_eval(abnormalities_str)
        return abnormalities_str
    
    elif step == 'synonyms':
        if "synonyms =" in string:
            start_index = string.rfind("synonyms =") + len("synonyms =")
        elif "synonyms=" in string:
            start_index = string.rfind("synonyms=") + len("synonyms=")
        elif "{" in string:
            start_index = string.find("{")
        else:
            return None
        end_index = string.rfind("}", start_index) + 1
        synonyms_str = string[start_index:end_index].strip()
        return synonyms_str
    
    elif step == 'longitudinal pancreas':
        first_diagnosis_pattern = r'first diagnosis report\s*[=:]\s*(\d+|none)(?=[;\n.]|$)'
        pre_diagnosis_pattern = r'pre-diagnosis reports\s*[=:]\s*([\d,]+|none)(?=[;\n.]|$)'

        first_diagnosis_match = re.search(first_diagnosis_pattern, string, re.IGNORECASE)
        pre_diagnosis_match = re.search(pre_diagnosis_pattern, string, re.IGNORECASE)

        return {
            'First Diagnosis Report': first_diagnosis_match.group(1) if first_diagnosis_match else None,
            'Pre-Diagnosis Reports': pre_diagnosis_match.group(1) if pre_diagnosis_match else None
        }
        
    elif step == 'longitudinal pancreas diagnosis':
        tumor_pattern = r"tumor types\s*:\s*(.*?)(?=$|\n)"
        match = re.search(tumor_pattern, string, re.IGNORECASE)
        if not match:
            return None  # no "tumor types:" line found
        
        # Extract the entire substring after "tumor types:"
        raw_tumor_str = match.group(1).strip()
        # Example raw_tumor_str might be "PDAC; Cyst; Unknown;" or "none;"
        
        return {'Tumor Types': raw_tumor_str}
        
    elif step == 'refine normal pancreas':
        # Case- and line-insensitive patterns
        _PATTERNS = {
            'Decision': re.compile(r'^\s*decision\s*[:=\-]\s*(exclude|include)', re.I | re.M),
            'Confidence': re.compile(r'^\s*confidence\s*[:=\-]\s*(high|medium|low)', re.I | re.M),
            'Human Review Needed': re.compile(r'^\s*human\s+review\s+required\s*[:=\-]\s*(yes|no)', re.I | re.M)
        }

        out = {}
        for key, pattern in _PATTERNS.items():
            m = pattern.search(string)
            out[key] = m.group(1).strip().capitalize() if m else None
        return out
    
    elif step == 'stage pancreas':
        # Extract the pancreas staging JSON (no_pancreatic_tumor + tumors list)
        return parse_pancreas_stage_json(string)
    
    elif step == 'refine normal pancreas 2':
        # Case- and line-insensitive pattern
        _PATTERNS = {
            # optional bullet ( - * • ) + optional spaces, then “decision”
            'Decision': re.compile(
                r'^[\s]*[-*•]?\s*decision\s*[:=\-]\s*(exclude|include)',
                re.I | re.M
            ),
        }

        out = {}
        for key, pattern in _PATTERNS.items():
            m = pattern.search(string)
            out[key] = m.group(1).strip().capitalize() if m else None
        return out 
    
    else:
        raise ValueError('Invalid step')
    
def get_random_examples(target,limit,num,data):
    examples=[]
    for i in range(num):
        example=random.randint(0,limit)
        while example==i:
            example=random.randint(0,data.shape[0])
        examples.append(example)
    return examples

def generate_metrics(data, dnn_outputs, id_column='Anon Acc #', columns_to_evaluate=None,MRNs=None,step='tumor detection'):
    """
    Generates and prints confusion matrices and evaluation metrics for specified columns in two DataFrames.
    
    Parameters:
    data (pd.DataFrame): DataFrame containing ground truth labels.
    dnn_outputs (pd.DataFrame): DataFrame containing predicted labels.
    id_column (str): The column name used to match rows between the DataFrames (default is 'Anon Acc #').
    columns_to_evaluate (list): List of column names to evaluate. If None, defaults to ['Liver Tumor', 'Kidney Tumor', 'Pancreas Tumor'].
    """

    if step=='malignancy detection':
        # Step 1: Create a new DataFrame with the selected columns
        dnn_outputs = copy.deepcopy(dnn_outputs[[id_column, 'Malignant Tumor in liver', 'Malignant Tumor in pancreas', 'Malignant Tumor in kidney']])
        # Step 2: Rename the columns
        dnn_outputs.columns = [id_column, 'Liver Tumor', 'Pancreas Tumor', 'Kidney Tumor']

    original_dnn_outputs=copy.deepcopy(dnn_outputs)
    original_data=copy.deepcopy(data)

    # Ensure both DataFrames are sorted by the identifier column
    data = data.sort_values(id_column).reset_index(drop=True)
    dnn_outputs = dnn_outputs.sort_values(id_column).reset_index(drop=True)

    #drop any row with nan in the dnn_outputs
    dnn_outputs=dnn_outputs.dropna()
    #check all rows in data and drop them if they are not present in dnn_outputs
    data=data[data[id_column].isin(dnn_outputs[id_column])]
    
    # Drop any row that has a MRN value not in the MRNs list
    if MRNs is not None:
        data = data[data[id_column].isin(MRNs)]
        dnn_outputs = dnn_outputs[dnn_outputs[id_column].isin(MRNs)]

    # Check if the identifier columns match
    if not data[id_column].equals(dnn_outputs[id_column]):
        raise ValueError(f"The '{id_column}' columns do not match between the DataFrames.")
    
    # Default columns to evaluate if not provided
    if columns_to_evaluate is None:
        columns_to_evaluate = ['Liver Tumor', 'Kidney Tumor', 'Pancreas Tumor']

    #print FPs and FNs
    for column in columns_to_evaluate:
        
        #drop in y_pred and y_true any row where y_true is nan
        data_no_nan=data.dropna(subset=[column])
        #get the same rows in dnn_outputs
        dnn_outputs_no_nan=dnn_outputs[dnn_outputs[id_column].isin(data_no_nan[id_column])]

        y_true = data_no_nan[column]
        y_pred = dnn_outputs_no_nan[column]

        # print False Positives and False Negatives
        print('False Positives and False Negatives for',column)
        for case in data_no_nan[(y_true==0) & (y_pred==1)][[id_column,column]].values:
            print('False Positives for',column,':',case[0])
            #print content of the report
            print(data_no_nan[data_no_nan[id_column]==case[0]]['Anon Report Text'].values[0])
            print(' \n ')
        for case in data_no_nan[(y_true==1) & (y_pred==0)][[id_column,column]].values:
            print('False Negatives for',column,':',case[0])
            #print content of the report
            print(data_no_nan[data_no_nan[id_column]==case[0]]['Anon Report Text'].values[0])
            print(' \n ')
    
    # Compute and print confusion matrices and metrics for each specified column
    for column in columns_to_evaluate:
        #drop in y_pred and y_true any row where y_true is nan
        data_no_nan=data.dropna(subset=[column])
        #get the same rows in dnn_outputs
        dnn_outputs_no_nan=dnn_outputs[dnn_outputs[id_column].isin(data_no_nan[id_column])]

        y_true = data_no_nan[column]
        y_pred = dnn_outputs_no_nan[column]

        print('Organ:',column)
        print('Gorund Truth:',y_true)
        print('Predictions:',y_pred)

        cm = confusion_matrix(y_true, y_pred,labels=[0,1])
        
        tn, fp, fn, tp = cm.ravel()
        
        # Manually calculate metrics, setting to NaN if division by zero occurs
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        ppv = tp / (tp + fp) if (tp + fp) > 0 else np.nan
        f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else np.nan
        
        print(f"Metrics for {column}:")
        print(f"Confusion Matrix:\n{cm}\n")
        print(f"True Positives (TP): {tp}")
        print(f"False Positives (FP): {fp}")
        print(f"True Negatives (TN): {tn}")
        print(f"False Negatives (FN): {fn}")
        print(f"Sensitivity: {sensitivity if not np.isnan(sensitivity) else 'NaN (0/0)'}")
        print(f"Specificity: {specificity if not np.isnan(specificity) else 'NaN (0/0)'}")
        print(f"PPV (Precision): {ppv if not np.isnan(ppv) else 'NaN (0/0)'}")
        print(f"F1-score: {f1 if not np.isnan(f1) else 'NaN (0/0)'}\n")

    #analyze nan cases
    for column in columns_to_evaluate:
        #print report for any row where original_dnn_output is nan and data is not
        j=0
        for case in original_dnn_outputs[original_dnn_outputs[column].isna()][id_column].values:
            #check if case is nan in original_data:
            if original_data[original_data[id_column]==case][column].isna().values[0]:
                continue
            if 'too small' in original_data[original_data[id_column]==case]['Anon Report Text'].values[0].lower():
                continue
            if 'ill-defined' in original_data[original_data[id_column]==case]['Anon Report Text'].values[0].lower():
                continue
            #print content of the report for case
            for i in list(range(4)):
                print('\n')
            if step=='tumor detection':
                print('NaN tumor detection case but tumor label not nan for',column,':',case)
            elif step=='malignancy detection':
                print('NaN malignancy detection case but label not nan for',column,':',case)
            print(original_data[original_data[id_column]==case]['Anon Report Text'].values[0])
            j+=1
        print('Total of',j,'cases with NaN in the predictions but not in the ground truth for',column)

def get_first_malignancy(accession_number, df, id_column='Accession Number'):
    """
    Function to get the Accession Number of the first malignancy diagnosis
    for a patient based on the given pre-diagnosis Accession Number using the 
    'pancreatic cancer timeline' column.
    
    Parameters:
    - accession_number: The Accession Number of the pre-diagnosis report
    - df: The dataframe containing the reports
    
    Returns:
    - The Accession Number of the first malignancy diagnosis for the patient, 
      or None if no malignancy is found.
    """
    
    # Get the patient based on the provided Accession Number
    patient_row = df[df[id_column] == accession_number]
    
    if patient_row.empty:
        raise ValueError(f"No report found with Accession Number {accession_number}")
    
    # Get the patient's Assigned Number
    assigned_number = patient_row['Assigned Number'].values[0]
    
    # Filter the reports for the same patient (Assigned Number)
    patient_reports = df[df['Assigned Number'] == assigned_number]
    
    # Sort the patient's reports by 'Exam Started Date' to ensure chronological order
    patient_reports = patient_reports.sort_values(by='Exam Started Date')
    
    # Find the first report where 'pancreatic cancer timeline' is 'first diagnosis'
    first_diagnosis_row = patient_reports[patient_reports['pancreatic cancer timeline'] == 'first positive'].head(1)
    
    if not first_diagnosis_row.empty:
        print(f"First malignancy diagnosis found for patient with Accession Number {accession_number}:")
        return first_diagnosis_row[id_column].values[0]
    else:
        raise ValueError(f"No first malignancy diagnosis found for patient with Accession Number {accession_number}")
            

def write_tumor_multi_rows(writer, sample, tumors, answer, multi_organ=False,report=None, step=None,max_rows=None):
    if step == 'stage pancreas':
        # Here, `tumors` is actually the full dict returned by parse_pancreas_stage_json:
        # {
        #   "no_pancreatic_tumor": bool,
        #   "no_pancreatic_tumor_justification": str,
        #   "no_pancreatic_tumor_supporting_quotes": [...],
        #   "tumors": [ {...}, {...}, ... ]
        # }
        data = tumors if isinstance(tumors, dict) else {}
        no_pancreatic_tumor = bool(data.get("no_pancreatic_tumor", False))
        tumor_list = data.get("tumors", []) or []

        # If no pancreatic tumor, or tumors list is empty -> single summary row
        if no_pancreatic_tumor or not tumor_list:
            row = [
                sample,
                "no pancreatic tumor",   # Tumor ID
                "pancreas",              # Organ
                "u",                     # Tumor Type
                "u",                     # Tumor Location
                "u",                     # Tumor Size (mm)
                "u",                     # Type Certainty
                "u",                     # Type Detail
                "unknown",               # SMA Encasement
                "unknown",               # CA Encasement
                "unknown",               # CHA Encasement
                "unknown",               # Aorta Encasement
                "unknown",               # SMV Encasement
                "unknown",               # MPV Encasement
                answer                   # DNN Answer
            ]
            if report is not None:
                row.append(report)
            writer.writerow(row)
            return

        # Otherwise, one row per pancreatic tumor
        for t in tumor_list:
            # safer access with dict.get
            tumor_index = t.get("tumor_index", None)
            tumor_id = f"pancreas tumor {tumor_index}" if tumor_index is not None else "pancreas tumor u"

            tumor_type = t.get("tumor_type", "u")
            type_detail = t.get("tumor_type_other_detail", "") or "u"
            type_certainty = t.get("type_certainty", "u")

            size_mm = t.get("size_mm", "")
            if not size_mm:
                size_str = "u"
            else:
                # parse_pancreas_stage_json already converted lists -> "a x b" strings
                size_str = str(size_mm)

            location = t.get("location", "unknown")

            vessels = t.get("vessel_encasement", {}) or {}
            sma   = vessels.get("SMA",   "unknown")
            ca    = vessels.get("CA",    "unknown")
            cha   = vessels.get("CHA",   "unknown")
            aorta = vessels.get("aorta", "unknown")
            smv   = vessels.get("SMV",   "unknown")
            mpv   = vessels.get("MPV",   "unknown")

            row = [
                sample,
                tumor_id,
                "pancreas",     # Organ always pancreas
                tumor_type,
                location,
                size_str,       # Tumor Size (mm) as "a x b"
                type_certainty,
                type_detail,
                sma,
                ca,
                cha,
                aorta,
                smv,
                mpv,
                answer
            ]
            if report is not None:
                row.append(report)
            writer.writerow(row)

        return
    # --- END of stage pancreas branch ---
    
    rows_written = 0
    for tumor_id, tumor_data in tumors.items():
        if max_rows is not None and rows_written >= max_rows:
            break
        rows_written += 1
        size = tumor_data.get('size', [])
        
        if isinstance(size, (float, int)):  # Single numeric value
            #check if nan
            if np.isnan(size):
                size_str='u'
            else:
                size_str = f"{size} mm"
        elif isinstance(size, list):  # List of numeric values
            size_str = " x ".join(map(str, size))
        elif size == 'multiple':  # Handle 'multiple' cases
            size_str = "multiple"
        else:  # Handle unexpected cases
            size_str = "U"

        if step=='HCC':
            #{'liver tumor 1': {'type': 'hcc', 'certainty': 'certain', 'size': nan, 'location': 'segment 6', 'arterial enhancement': 'absent', 'washout': 'absent', 'capsule': 'u', 'threshold growth': 'u', 'LI-RADS': 'lr-tr nonviable'}}

            # Build the output row based on the step.
            # For HCC, we require all fields.
            row = [
                sample,
                tumor_id,
                tumor_data.get('type', 'U'),
                tumor_data.get('certainty', 'U'),
                size_str,
                tumor_data.get('location', 'U'),
                tumor_data.get('arterial enhancement', 'U'),
                tumor_data.get('washout', 'U'),
                tumor_data.get('capsule', 'U'),
                tumor_data.get('threshold growth', 'U'),
                tumor_data.get('LI-RADS', 'U'),
                answer
            ]
            if report is not None:
                row.append(report)
        else:
            row = [
                sample,
                tumor_id,
                tumor_data.get('organ', np.nan),
                tumor_data.get('type', np.nan),
                tumor_data.get('location', np.nan),
                size_str,
                tumor_data.get('attenuation', np.nan),
                tumor_data.get('certainty', np.nan),
                answer  # Add the raw LLM answer to the row
            ]
        if report is not None:
            row.append(report)
        writer.writerow(row)

def append_no_lesion_rows(
    data: pd.DataFrame,
    accession: str,
    header: list[str],
    csv_path: str,
    accn_col: str = "Encrypted Accession Number",
    no_lesion_col: str = "no lesion",
):
    """
    Copy all tumour rows for `accession` to `csv_path`.
    If the 'no lesion' flag is True for every row, Series / Image / DNN
    fields are written as 'u'.
    """

    rows = data.loc[data[accn_col] == accession].copy()
    if rows.empty:
        raise ValueError(f"No rows for accession '{accession}'")

    if rows[no_lesion_col].all():
        rows["Series"] = "u"
        rows["Image"]  = "u"
        rows["DNN answer Se/Im"] = "u"
    else:
        raise ValueError(
            "append_no_lesion_rows should only be called when every row has "
            "'no lesion' == True"
        )

    #write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
    with open(csv_path, "a", newline="") as f:
        writer = csv.writer(f)

        #if write_header:
        #    writer.writerow(header)

        for _, r in rows.iterrows():
            csv_row = [r.get(col, np.nan) for col in header]
            writer.writerow(csv_row)
            
RISK_JSON_KEYS = [
    # Core demographics / anthropometrics
    "smoking",
    "obesity",
    "patient_weight_kg",
    "patient_height_cm",
    "bmi",

    # Clinical risk factors
    "alcohol_use",
    "chronic_pancreatitis",
    "acute_pancreatitis",
    "pancreatitis",
    "family_history_pancreatic_cancer",

    # Genetics
    "high_risk_germline_mutation",
    "high_risk_mutation_list",

    # Symptoms / conditions
    "diabetes",
    "weight_loss",

    # Cholestasis (split)
    "painless_jaundice",
    "pruritus",
    "dark_urine",
    "pale_acholic_stools",

    # Symptom cluster (split)
    "epigastric_pain",
    "back_pain",
    "anorexia",
    "steatorrhea",

    "dvt_or_pe",
    "abdominal_pain",

    # LFTs
    "high_direct_bilirubin_or_alp",
    "direct_bilirubin_value",
    "direct_bilirubin_unit",
    "alp_value",
    "alp_unit",

    # CA 19-9
    "elevated_ca19_9",
    "ca19_9_value",
    "ca19_9_unit",

    # Inflammation / nutrition
    "high_nlr_or_thrombocytosis_or_low_albumin",
    "nlr",
    "anc_value",
    "anc_unit",
    "alc_value",
    "alc_unit",
    "platelets_value",
    "platelets_unit",
    "albumin_value",
    "albumin_unit",

    # Imaging
    "pancreatic_duct_dilatation",
    "pancreatic_focal_atrophy_or_contour_change",
    "ipmn",
    "pancreatic_cyst",
]

PATHOLOGY_JSON_KEYS = [
    "patient_has_malignant_tumor",
    "malignant_organs",
    "metastasis_organs",
    "benign_mass_or_cyst",
    "tumor_types"
]

def inference_loop(data, base_url='http://0.0.0.0:8888/v1', step='tumor detection', outputs={}, examples=0, fast=True,
                   institution='UCSF', save_name=None, restart=False,item_list=None,max_rows=None):
    """
    ### Function Documentation: 'inference_loop'
    
    #### Summary:
    This function processes medical data, performs inference on tumor detection or classification tasks, and saves or updates results. It supports various steps, such as tumor detection, malignancy detection, and type/size analysis, and includes logic to handle different data structures and institutions.
    
    #### Parameters:
    - **data** ('pd.DataFrame'): Input dataset containing radiology/pathology records.
    - **base_url** ('str'): URL of the inference API, the 4 numbers at the end are the ones you used in VLLM serve. Default is ''http://0.0.0.0:8888/v1''. 
    - **step** ('str'): Task to perform (e.g., ''tumor detection'', ''malignancy detection'', ''type and size', 'type and size pathology').
    - **outputs** ('dict' or 'list'): Previous results to be updated. Defaults to an empty dictionary.
    - **examples** ('int'): Number of examples for contextual inference. Default is '0'.
    - **fast** ('bool'): Flag to enable fast processing by reducing prompt size. Not available to all steps. Reduces accuracy. Default is 'True'.
    - **institution** ('str'): Institution name, which affects column naming and processing logic. Default is ''UCSF'', can be COH.
    - **save_name** ('str'): Name of the CSV file to save results. Default is 'None'.
    - **restart** ('bool'): Whether to restart processing by clearing saved results. Default is 'False'. Careful.
    - **item_list** ('list'): List of specific items to process. Default is 'None'.
    
    #### Returns:
    - **'pd.DataFrame'**: Updated outputs as a DataFrame containing processed results.
    
    #### Steps:
    1. **Institution-Specific Initialization**:
       - Sets column names and organ list based on the institution.
       - Validates the presence of necessary columns.
    
    2. **Save File Handling**:
       - Initializes or reads a save file if specified.
       - Manages restarting logic by clearing or creating the file.
    
    3. **Outputs Preparation**:
       - Converts 'outputs' to a dictionary if necessary.
       - Handles duplicate entries.
    
    4. **Processing Loop**:
       - Iterates over organs (if applicable) and rows of the dataset.
       - Skips already processed or irrelevant samples.
       - Retrieves relevant report text for inference.
       - Selects examples if needed.
    
    5. **Inference and Interpretation**:
       - Sends data to an inference API.
       - Interprets and updates outputs based on the specified step.
    
    6. **Saving Results**:
       - Appends results to the specified save file in the appropriate format.
    
    7. **Return Results**:
       - Returns updated outputs as a DataFrame.
    
    #### Supported Steps:
    - ''tumor detection'': Detects tumors across specified organs.
    - ''malignancy detection'': Identifies malignancy in detected tumors.
    - ''malignant size'': Determines the size of malignant tumors.
    - ''type and size'': Analyzes tumor type and size.
    - ''type and size multi-organ'': Multi-organ tumor type and size analysis.
    - ''diagnoses'': Adds abnormality and inference results to the dataset.
    - ''time machine'': Performs longitudinal analysis using future reports.
    
    #### Example Usage:
    '''python
    # Example: Tumor detection with a save file
    results = inference_loop(
        data=my_data,
        base_url='http://127.0.0.1:8000',
        step='tumor detection',
        save_name='tumor_detection_results.csv',
        restart=False
    )
    """
    outputs = copy.deepcopy(outputs)
    
    error_tolerance = 3  # Number of allowed consecutive errors before stopping
    error_count = 0    # Counter for consecutive errors
    
    if step == 'find matching reports':
        data = data.sort_values(by='similarity', ascending=False)

    if institution == 'UCSF':
        id_column = 'Anon Acc #'
        #check if the columns are present
        if id_column not in data.columns:
            id_column='Acc #'
        if id_column not in data.columns:
            id_column='id'
        if id_column not in data.columns:
            id_column='Encrypted Accession Number'
        if id_column not in data.columns:
            id_column='Accession Number'
        if id_column not in data.columns:
            id_column='accessionnumber'
        if id_column not in data.columns:
            id_column='BDMAP_ID'
        if id_column not in data.columns:
            id_column='BDMAP ID'
        organs = ['liver', 'kidney', 'pancreas']
        report_column = 'Anon Report Text'
        if report_column not in data.columns:
            report_column='Findings'
        if report_column not in data.columns:
            report_column='Report Text'
        if report_column not in data.columns:
            report_column='Report'
        if report_column not in data.columns:
            report_column='report'
        if report_column not in data.columns:
            report_column='answer'
        if step == 'tumor slices':
            organs = ['pancreas']
            id_column = 'Encrypted Accession Number'
            report_column = 'Findings'
            if data[report_column].isna().sum() > 0:
                #fill nans with Report
                data[report_column] = data[report_column].fillna(data['Report'])
            if data[report_column].isna().sum() > 0:
                raise ValueError('There is nan in the Findings and Report columns (aligned)')
        if step=='HCC':
            organs=['liver']
        if step=='longitudinal pancreas' or step=='longitudinal pancreas diagnosis':
            organs=['pancreas']
            id_column='Encrypted Accession Number'
        if step=='find matching reports':
            organs = ['pancreas']
            id_column = 'deid_note_key'
            #report_column = 'note_text'
            report_column = 'findings_really'
        if step=='pre-diagnostic confirmation':
            organs=['pancreas']
            report_column='oldest pre-diagnostic report'
            if report_column not in data.columns:
                report_column='Report'
                #report_column = 'note_text'
            id_column='Patient ID'
            if id_column not in data.columns:
                #id_column='Encrypted Accession Number'
                id_column = 'deid_note_key'
        labels = True
        if step=='risk factors pancreas':
            organs=['pancreas']
            report_column = 'note_text'
            id_column = 'row_identifier'
        if step=='pathology':
            organs=['pancreas']
            report_column = 'note_text'
            id_column = 'deid_note_key'
        if step=='pathology many reports':
            organs=['pancreas']
            report_column = 'pathology_notes_all'
            id_column = 'Patient ID'
        if step=='refine normal pancreas':
            organs=['pancreas']
            if report_column=='Findings':
                raise ValueError('Findings not allowed for refine normal pancreas, use full report')
        if step=='refine normal pancreas 2':
            organs=['pancreas']
            if report_column=='Findings':
                raise ValueError('Findings not allowed for refine normal pancreas, use full report')
        if step=='stage pancreas':
            organs=['pancreas']
        if report_column not in data.columns:
            raise ValueError('No report column found')
        if id_column not in data.columns:
            raise ValueError('No ID column found')
    else:
        id_column = 'Accession Number'
        organs = ['pancreas']
        report_column = 'Report Text'
        labels = False

    if step == 'tumor detection' or step == 'diagnoses':
        lp=['']#no need to loop multiple times
    else:
        lp=organs

    saved = set()  # Use a set to store processed samples for fast lookup

    if step == 'diagnoses':
        header=data.columns.tolist()+['Abnormalities', 'DNN answer']
    if step=='find matching reports':
        header = data.columns.tolist()+['Matching Reports', 'DNN answer']
    if step == 'tumor slices':
        header = data.columns.tolist()+['Series', 'Image', 'DNN answer Se/Im']
    if step=='risk factors pancreas':
        header=data.columns.tolist()+RISK_JSON_KEYS+ ['DNN answer']
    if step=='pathology':
        header=data.columns.tolist()+PATHOLOGY_JSON_KEYS+ ['DNN answer']
    if step=='pathology many reports':
        header=data.columns.tolist() + ["organs_with_primary_malignant_tumors",
            "organs_with_metastatic_tumors",
            "organs_with_benign_tumors",
            "organs_with_unknown_tumor_type",
            "organs_with_unknown_tumor_type",
            'DNN answer']

    # Check if the save file exists and restart is False
    if save_name is not None:
        if save_name[-4:]=='.csv':
            save_name=save_name[:-4]
        if step.replace(' ', '_') not in save_name:
            save_name = save_name + '_' + step.replace(' ', '_')
        save_name+='.csv'
        file_path = save_name
        print('Save path:',file_path)

        if os.path.exists(file_path) and not restart:
            # Check if file is not empty
            if os.stat(file_path).st_size > 0:
                # Read the first column (Anon Acc # or Accession Number) to skip already processed samples
                print(pd.read_csv(file_path))
                #raise ValueError('Save not implemented')
                saved = set(pd.read_csv(file_path)[id_column].tolist())
                #saved = set(pd.read_csv(file_path).iloc[:, 0].tolist())
            else:
                saved = set()
        else:
            saved = set()
        
        print('Number of saved samples:', len(saved))

        # If restart is True, delete the existing file and reset
        if restart and os.path.exists(file_path):
            os.remove(file_path)
            

        # If the file does not exist or was deleted, create it and write headers
        if not os.path.exists(file_path):
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                if step == 'tumor detection':
                    writer.writerow([id_column, 'Liver Tumor', 'Kidney Tumor', 'Pancreas Tumor', 'DNN answer'])
                elif step=='pre-diagnostic confirmation':
                    writer.writerow([id_column, 'Pancreatic Tumor Suspicion', 'Pancreas Surgery ', 'Cancer History', 'report', 'DNN answer'])
                elif step=='find matching reports':
                    writer.writerow(header)
                elif step == 'malignancy detection':
                    writer.writerow([id_column, 'Liver Tumor', 'Kidney Tumor', 'Pancreas Tumor', 'DNN answer', 'Malignant Tumor in '+organs[0], 'DNN answer 2'])
                elif step == 'malignant size':
                    writer.writerow([id_column, 'Liver Tumor', 'Kidney Tumor', 'Pancreas Tumor', 'DNN answer', 'Malignant Tumor in '+organs[0], 'DNN answer 2', 'Size of Largest Malignant Tumor in '+organs[0], 'DNN answer 3'])
                elif step == 'time machine':#use information from future report to understand if past report shows a malignant tumor
                    writer.writerow([id_column, 'Liver Tumor', 'Kidney Tumor', 'Pancreas Tumor', 'DNN answer', 'Malignant Tumor in '+organs[0], 'DNN answer 2', 'Size of Largest Malignant Tumor in '+organs[0], 'DNN answer 3', 'Longitudinal Analysis: Very Likely Malignancy in '+organs[0], 'Longitudinal Analysis: Size of Very Likely Largest Malignant Tumor in '+organs[0], 'DNN answer 4'])
                elif step == 'type and size' or step=='type and size pathology':
                    writer.writerow([id_column, "Tumor ID", "Tumor Type", "Type Certainty", "Tumor Size", "Tumor Location"])
                elif step == 'pathology' or step=='pathology many reports':
                    writer.writerow(header)
                elif step == 'type and size multi-organ':
                    writer.writerow([id_column, "Tumor ID", "Organ", "Tumor Type", "Tumor Location", "Tumor Size (mm)", "Tumor Attenuation", "Type Certainty","DNN Answer","Report"])
                elif step == "stage pancreas":
                    writer.writerow([id_column, "Tumor ID", "Organ", "Tumor Type", "Tumor Location", "Tumor Size (mm)", "Type Certainty", "Type Detail", "SMA Encasement", "CA Encasement", "CHA Encasement", "Aorta Encasement", "SMV Encasement", "MPV Encasement", "DNN Answer","Report"])#Organ will be always pancreas
                elif step == 'HCC':
                    writer.writerow([id_column, "Tumor ID", "Tumor Type", "Type Certainty", "Tumor Size", "Tumor Location", "Arterial Enhancement", "Washout", "Capsule", "Threshold Growth", "LI-RADS","DNN Answer","Report"])
                elif step == 'diagnoses':
                    header=data.columns.tolist()+['Abnormalities', 'DNN answer']
                    writer.writerow(header)
                elif step == 'longitudinal pancreas':
                    writer.writerow([id_column, "First Diagnosis", "Pre-diagnosis", "Ordered Accessions","reports","DNN Answer"])
                elif step == 'longitudinal pancreas diagnosis':
                    writer.writerow([id_column, "Tumor Types","reports","DNN Answer"])
                elif step =='tumor slices':
                    writer.writerow(header)
                elif step=='refine normal pancreas':
                    writer.writerow([id_column, 'Decision', 'Confidence', 'Human Review Needed', "Report", 'DNN answer'])
                elif step=='refine normal pancreas 2':
                    writer.writerow([id_column, 'Decision', "Report", 'DNN answer'])
                elif step=='risk factors pancreas':
                    writer.writerow(header)
                    
    # Convert outputs to dictionary if needed
    if isinstance(outputs, list):
        pass
    elif not isinstance(outputs, dict):
        # Check for duplicates and handle them
        if outputs[id_column].duplicated().any():
            print(f"Warning: There are duplicate values in the {id_column} column. We are taking only the first occurrence.")
            raise ValueError('Duplicated values in the id column')
            outputs = outputs.drop_duplicates(subset=id_column)

       #print('outputs:',outputs)
        outputs = outputs.set_index(id_column).to_dict(orient='index')

    if step == 'type and size' or step == 'type and size multi-organ' or step=='type and size pathology' or step=="stage pancreas":
        old_step=copy.deepcopy(outputs)
        outputs = {}
        

    # Loop over organs and data
    if step == 'longitudinal pancreas' or step == 'longitudinal pancreas diagnosis':
        #here, we loop over patients, not over reports
        cases = data['Patient ID'].unique().tolist()
    elif step == 'tumor slices':
        cases = data['Encrypted Accession Number'].unique().tolist()
    else:
        cases = range(data.shape[0])
    
    for organ in lp:
        for i in cases:
            start=time.time()
            if step == 'tumor slices':
                row = data[data['Encrypted Accession Number']==i]
                sample = i
                try:
                    if row.iloc[0]['no lesion']:
                        append_no_lesion_rows(data,i,header,file_path)
                except:
                    raise ValueError('The no lesion column is needed for tumor slices step')
                    
                row = row.head(1)#get first row
                try:
                    report = row['Findings'].values[0]
                except:
                    # 1) Show every column
                    pd.set_option("display.max_columns", None)
                    # 2) Prevent line‑wrapping (so it doesn’t insert “...”)
                    pd.set_option("display.width", None)
                    # 3) Show full cell contents (no truncation of long strings)
                    pd.set_option("display.max_colwidth", None)
                    # Now print your filtered DataFrame:
                    print(data[data["Encrypted Accession Number"] == i])
                    report = row['Findings'].values[0]
                    
                if item_list is not None:
                    if sample not in item_list:
                        continue
                
            elif step != 'longitudinal pancreas' and step != 'longitudinal pancreas diagnosis':
                sample = data.iloc[i][id_column]
                report,_=get_report_n_label(data,i,row_name=report_column,get_date=False,id_col=id_column) 

                # Skip if the sample is in the saved list
                if sample in saved:
                    print(f'Skipping sample, already saved: {sample}')
                    continue
                if item_list is not None:
                    if sample not in item_list:
                        continue
                else:
                    if step == 'malignancy detection':
                        # Skip if the outputs do show no tumor in the organ
                        if outputs[sample][organ.capitalize()+' Tumor'] != 1.0:
                            print(f'Skipping sample, no certain tumor in {organ}: {sample}')
                            continue

                    if step == 'type and size' or step=='type and size pathology':
                        if isinstance(old_step, dict):
                            # Skip if the outputs do show no tumor in the organ
                            if sample in old_step and old_step[sample][organ.capitalize()+' Tumor'] != 1.0:
                                print(f'Skipping sample, no certain tumor in {organ}: {sample}')
                                continue
                        elif isinstance(old_step, list):
                            if sample not in old_step:
                                print(f'Skipping sample, no certain tumor in {organ}: {sample}')
                                continue

                    if step == 'malignant size':
                        if sample not in outputs:
                            print(f'Skipping sample, not yet predicted for malignancy: {sample}')
                            continue
                        # Skip if the outputs do show no tumor in the organ
                        if outputs[sample][organ.capitalize()+' Tumor'] != 1.0 or outputs[sample]['Malignant Tumor in '+organ] != 1.0:#measuring certain and uncertain tumors
                            print(f'Skipping sample, no certain malignant tumor in {organ}: {sample}')
                            continue

            if step == 'time machine':
                print(data.iloc[i]['pancreatic cancer timeline'])
                #check if data.iloc[i]['pancreatic cancer timeline'] is nan
                if not isinstance(data.iloc[i]['pancreatic cancer timeline'],str):
                    print(f'Skipping sample, no pre-diagnosis report: {sample}')
                    continue
                elif 'pre-diagnosis' not in data.iloc[i]['pancreatic cancer timeline']:
                    print(f'Skipping sample, no pre-diagnosis report: {sample}')
                    continue
                
                #get report of first diagnosis
                first_diagnosis=get_first_malignancy(sample, data, id_column=id_column)

                print('First diagnosis:',first_diagnosis)
            else:
                first_diagnosis=None
                
            if step == 'longitudinal pancreas' or step == 'longitudinal pancreas diagnosis':
                sample = i
                _,reports = get_longitudinal_reports(data, sample)
                #concat list of strings
                reports = '\n\n'.join(reports)
                print(f"Processing patient {sample}...")



            # Example selection logic
            if examples == 0:
                ex = []
            else:
                if institution != 'UCSF':
                    raise ValueError('Only UCSF institution is supported for examples')
                ex = get_random_examples(target=i, num=examples, limit=data.shape[0] - 1, data=data, step=step, organ=organ, outputs=outputs)


            try:
                answer = run(target=i, examples=ex, data=data, print_message=False, base_url=base_url, step=step, organ=organ, fast=fast,
                            row_name=report_column, id_column=id_column,future_report=first_diagnosis)
                error_count = 0  # Reset error count on successful inference
            except:
                error_count += 1
                print(f"Error encountered for sample {sample}. Consecutive error count: {error_count}")
                if error_count > error_tolerance:
                    print("Error tolerance exceeded. Running it anyway.")
                    answer = run(target=i, examples=ex, data=data, print_message=False, base_url=base_url, step=step, organ=organ, fast=fast,
                            row_name=report_column, id_column=id_column,future_report=first_diagnosis)
                else:
                    continue
            
            
            if step == 'longitudinal pancreas' or step == 'longitudinal pancreas diagnosis':
                answer,accessions=answer
            
            # Interpret the output
            print('Arrived here')
            print('Answer:',answer)

            out = interpret_output(answer, step=step, organ=organ)
            
            
            print('Out:',out, flush=True)
            #raise ValueError(f'Out is: \n\n\n {out}')
            

            # Update the outputs based on the step
            if step == 'refine normal pancreas' or step == 'refine normal pancreas 2' or step == 'tumor detection' or \
                step == 'type and size' or step == 'type and size multi-organ' or step=='type and size pathology' or \
                    step == 'HCC' or step == 'longitudinal pancreas' or step == 'longitudinal pancreas diagnosis' or \
                        step == 'pre-diagnostic confirmation' or step=='stage pancreas':
                outputs[sample] = out
            elif step == 'malignancy detection' or step == 'malignant size' or step == 'time machine':
                #print('Outputs:',outputs)
                outputs[sample].update(out)
            elif step=='diagnoses':
                row=data.iloc[i].to_dict()
                row['Abnormalities']=out
                row['DNN answer']=answer
                outputs[sample]=row
            elif step=='find matching reports':
                row=data.iloc[i].to_dict()
                row['Matching Reports'] = out['Matching Reports']
                row['DNN answer']=answer
                outputs[sample]=row
            elif step == 'risk factors pancreas' or step=='pathology' or step=='pathology many reports':
                # Merge source row + clean JSON so header lookup works for all keys
                src = data.iloc[i].to_dict()
                src.update(out)  # 'out' is the clean risk dict now
                outputs[sample] = src
            
                    
                
            
                
            if step == 'diagnoses' or step == 'find matching reports':
                if save_name is not None:
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        print('Outputs:',outputs[sample])
                        tmp=[]
                        for h in header:
                            tmp.append(outputs[sample][h])
                        writer.writerow(tmp)
            elif step=='refine normal pancreas':
                if save_name is not None:
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        print('Outputs:',outputs[sample])
                        writer.writerow([sample]+[outputs[sample]['Decision'],outputs[sample]['Confidence'],outputs[sample]['Human Review Needed'],report,answer])
            elif step=='refine normal pancreas 2':
                if save_name is not None:
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        print('Outputs:',outputs[sample])
                        writer.writerow([sample]+[outputs[sample]['Decision'],report,answer])
            elif step == 'tumor slices':
                # select tumours for this accession
                accession = i
                subset = data[data['Encrypted Accession Number'] == accession].copy()
                if subset.empty:
                    raise ValueError(f"No rows for accession '{accession}'")

                # map Series / Image by tumour id
                series_map = out.set_index("Tumor ID")["Series"].to_dict()
                image_map  = out.set_index("Tumor ID")["Image"].to_dict()

                # prepare rows for writing
                rows_for_csv = []
                for _, row in subset.iterrows():
                    t_id = row['Tumor ID']

                    series_val = series_map.get(t_id, np.nan)
                    image_val  = image_map.get(t_id, np.nan)

                    # convert NaN to 'u' for CSV
                    series_val = "u" if pd.isna(series_val) else int(series_val)
                    image_val  = "u" if pd.isna(image_val)  else int(image_val)

                    # build row in header order
                    csv_row = []
                    for col in header:
                        if col == "Series":
                            csv_row.append(series_val)
                        elif col == "Image":
                            csv_row.append(image_val)
                        elif col == "DNN answer Se/Im":
                            csv_row.append(answer)
                        else:
                            csv_row.append(row[col])
                    rows_for_csv.append(csv_row)

                # write / append
                #write_header = not os.path.exists(csv_path) or os.path.getsize(csv_path) == 0
                with open(file_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    #if write_header:
                    #    writer.writerow(header)
                    writer.writerows(rows_for_csv)
            elif step == 'longitudinal pancreas':
                if save_name is not None:
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        print('Outputs:',outputs[sample])
                        writer.writerow([sample]+[outputs[sample]['First Diagnosis Report'],outputs[sample]['Pre-Diagnosis Reports'],accessions,reports,answer])
            elif step == 'pre-diagnostic confirmation':
                if save_name is not None:
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        print('Outputs:',outputs[sample])
                        writer.writerow([sample]+[outputs[sample]['Pancreatic Tumor Suspicion'],outputs[sample]['Pancreas Surgery'],outputs[sample]['Cancer History'],report,answer])
                else:
                    raise ValueError('not saving')
            elif step == 'longitudinal pancreas diagnosis':
                if save_name is not None:
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        print('Outputs:',outputs[sample])
                        writer.writerow([sample]+[outputs[sample]['Tumor Types'],reports,answer])
            elif step == 'risk factors pancreas' or step=='pathology':
                if save_name is not None:
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        print('Outputs:', outputs[sample])
                        outputs[sample]['DNN answer'] = answer   # put answer under the header key
                        for key in ("malignant_organs", "metastasis_organs", "benign_mass_or_cyst", "tumor_types"):
                            val = outputs[sample].get(key)
                            if isinstance(val, (list, dict)):
                                outputs[sample][key] = json.dumps(val, ensure_ascii=False)
                        row = [outputs[sample].get(h, None) for h in header]
                        writer.writerow(row)
            elif step=='pathology many reports':
                if save_name is not None:
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        print('Outputs:', outputs[sample])
                        outputs[sample]['DNN answer'] = answer   # put answer under the header key
                        for key in ("organs_with_primary_malignant_tumors",
                                    "organs_with_metastatic_tumors",
                                    "organs_with_benign_tumors",
                                    "organs_with_unknown_tumor_type"):
                            val = outputs[sample].get(key)
                            if isinstance(val, (list, dict)):
                                outputs[sample][key] = json.dumps(val, ensure_ascii=False)
                        row = [outputs[sample].get(h, None) for h in header]
                        writer.writerow(row)
            elif step == 'type and size' or step=='type and size pathology' or step == 'HCC':
                 with open(file_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    print('Outputs:',outputs[sample])
                    write_tumor_multi_rows(writer, sample, outputs[sample], answer,step=step,report=report)
            elif step == 'type and size multi-organ':
                with open(file_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    print('Outputs:',outputs[sample])
                    write_tumor_multi_rows(writer, sample, outputs[sample], answer, multi_organ=True,report=report,
                                           max_rows=max_rows)
                    
            elif step == 'stage pancreas':
                with open(file_path, 'a', newline='') as file:
                    writer = csv.writer(file)
                    print('Outputs:',outputs[sample])
                    write_tumor_multi_rows(writer, sample, outputs[sample], answer, multi_organ=True,report=report, step='stage pancreas')
                    
            elif step != 'type and size' and step != 'type and size multi-organ' and step!='type and size pathology' and step != 'HCC':
                # Append the new result to the CSV
                if save_name is not None:
                    with open(file_path, 'a', newline='') as file:
                        writer = csv.writer(file)
                        print('Outputs:',outputs[sample])
                        writer.writerow([sample] + list(outputs[sample].values()) + [answer])
                    

            print(f"Processed sample {sample} in {time.time() - start:.2f} seconds")

    if step != 'type and size' and step != 'type and size multi-organ' and step!='type and size pathology' and step != 'HCC':
        # Return updated outputs as a DataFrame
        outputs = pd.DataFrame.from_dict(outputs, orient='index')
        outputs.reset_index(inplace=True)
        outputs.rename(columns={'index': id_column}, inplace=True)

        # If institution is UCSF, generate metrics
        #if institution == 'UCSF' and 'size' not in step:
        #    generate_metrics(data, outputs, step=step)

    return outputs



# extract abnormal findings

abnormality_prompt_v0="""Your task is to extract and list all abnormal findings from a CT report provided below. 
Desired output format: provide me a python list of python dictionaries. Each dictionary should refer to an abnormality in the report. Each dictionary should have the following keys: abnormality, organ, location inside organ, size, and certainty. If you cannot infer any of these characteristics from the report, leave them as None.
Consider the following guidelines:
1- Abnormalities can be image findings, like lesions or ground-glass opacity or lesions, or diagnoses, like pneumonia or cancer.
2- If the report mentions a finding, but does not specify whether it is normal or abnormal, assume it is abnormal.
3- If a report presents both image findings and diagnoses, list them separately, even if they refer to the same abnormality.
4- If the report mentions no abnormalities, return an empty list.

"""

abnormality_prompt = """
Your task is to extract and list all **abnormal findings** from the CT report provided below ("CT report to analyze").  

**Output Format**  
Return a **Python list of dictionaries**, name it "abnormalities". Each dictionary should represent one abnormality and include the following keys:  
- **abnormality**:  type abnormality (e.g., ground-glass opacity, lesion, pneumonia). Avoid descriptions here. For example, use "lesion" instead of "large hypodense lesion". Be susinct and use the most standard terminology (in the singular), if possible.
- **organ**: The organ where the abnormality is found (e.g., lung, liver).  
- **location_inside_organ**: Specific location within the organ, if provided (e.g., upper lobe, segment VI).  
- **size**: The size of the abnormality, if mentioned (e.g., 2.5 cm).  
- **certainty**: Level of certainty mentioned in the report, characterize as high, medium or low. If nothing suggestest uncertainty, consider it high. 
- **description**: All the report's sentences related to the abnormality. The sentences should be **directly copied** from the report, just remove any personal name, patient MRN or Accession Number, if present. Copy all the sentences that refer to the abnormality or are even remotely related to it, even if they are not contiguous.

If organ, location_inside_organ, description or size cannot be inferred from the report, leave it as 'None'.

**Guidelines**  
1. **Types of Abnormalities**: Abnormalities can be **image findings** (e.g., lesions, ground-glass opacity) or **diagnoses** (e.g., pneumonia, cancer).  
2. **Assume Abnormal**: If the report mentions a finding but does not clarify whether it is normal or abnormal, assume it is **abnormal**.  
3. **Separate Listings**: If a report mentions both **image findings** and **diagnoses** referring to the same abnormality, list them **separately** as distinct entries.
4. **Lesion**: Do not use very generic terms as lesion if a more specific term is available (e.g., cyst, nodule, mass, trauma).
5. **No Abnormalities**: If the report mentions no abnormalities, return an **empty list**. "Unremarkable" findings are not considered abnormalities.
6. **Organ**: Use the most common organ name (e.g., “lungs” instead of “pulmonary parenchyma”). Avoid system names or very broad terms like “GI tract”, “reproductive organs” or "abdomen". Instead, use specific terms like “esophagus” or “uterus.”


**Example Input**  
CT Report:  
"There is a 2.5 cm hypodense lesion in segment VI of the liver involving the hepatic artery. Findings are suspicious for metastasis. Bilateral ground-glass opacities are noted in the lungs."

**Example Output**  
abnormalities = [
    {"abnormality": "lesion", "organ": "liver", "location_inside_organ": "segment VI", "size": "2.5 cm", "certainty": "high", "description": "There is a 2.5 cm hypodense lesion in segment VI of the liver involving the hepatic artery. Findings are suspicious for metastasis."},  
    {"abnormality": "ground-glass opacities", "organ": "lungs", "location_inside_organ": "bilateral", "size": None, "certainty": "high", "description": "Bilateral ground-glass opacities are noted in the lungs."},  
    {"abnormality": "metastasis", "organ": "liver", "location_inside_organ": None, "size": None, "certainty": "medium", "description": "There is a 2.5 cm hypodense lesion in segment VI of the liver involving the hepatic artery. Findings are suspicious for metastasis."}  
]

**Example Input**
CT Report:
Visualized lung bases: For chest findings, please see the separately dictated report from the CT of the chest of the same date.
Liver: Unremarkable
Gallbladder: Unremarkable
Spleen:  Unremarkable
Pancreas:  Unremarkable
Adrenal Glands:  Unremarkable
Kidneys:  Unremarkable
GI Tract:  Scattered colonic diverticula without evidence of diverticulitis.
Vasculature:  Unremarkable
Lymphadenopathy: Absent
Peritoneum: No ascites
Bladder: Unremarkable
Reproductive organs: Reproductive organs are surgically absent. Left pelvic sidewall multiloculated cystic lesion measures 3.5 x 2.2 cm on series 5 image 134. This was unchanged when compared to prior outside MRI from 10/12/2015.
Bones:  No suspicious lesions
Extraperitoneal soft tissues: Unremarkable
Lines/drains/medical devices: None
Impressions:
Joseph Hopkins has scattered colonic diverticula and multiloculated lesion on the left pelvic sidewall, unchanged from prior MRI.

**Example Output** 
abnormalities = [
    {"abnormality": "diverticulum", 
     "organ": "colon", 
     "location_inside_organ": None, 
     "size": None, 
     "certainty": "high",
     "description": "GI Tract:  [name removed] has scattered colonic diverticula without evidence of diverticulitis. Scattered colonic diverticula and multiloculated lesion on the left pelvic sidewall, unchanged from prior MRI."},
    
    {"abnormality": "cyst", 
     "organ": "pelvis", 
     "location_inside_organ": "left pelvic sidewall", 
     "size": "3.5 x 2.2 cm", 
     "certainty": "high",
     "description": Left pelvic sidewall multiloculated cystic lesion measures 3.5 x 2.2 cm on series 5 image 134. This was unchanged when compared to prior outside MRI from 10/12/2015. Scattered colonic diverticula and multiloculated lesion on the left pelvic sidewall, unchanged from prior MRI."}
]

CT report to analyze:
"""

group_synonyms = """I will provide you a list of diseases and findings taken from CT scan reports. Some of the names are synonyms or abbreviations of the same disease/finding. Your task is to group them together.
Output format: provide me a python dictionary where each key is a group of synonyms and the value is a list of all the synonyms in that group. You can take as key the most usual term in the group."""





def get_diagnoses(diagnoses_csv):
    # Load the CSV file
    diag = pd.read_csv(diagnoses_csv)
    diagnoses=[]
    errors=0
    for item in diag['Abnormalities'].str.replace('\n','').to_list():
        try:
            x=ast.literal_eval(item)
        except:
            errors+=1
            continue
        if type(x)==list and len(x)>0:
            for item in x:
                try:
                    diagnoses.append(item['abnormality'])
                except:
                    errors+=1

    print('Errors:',errors)

    return list(set(diagnoses))





#analyze outputs

def load(path):
    df=pd.read_csv(path)
    df=df.drop_duplicates(subset=['id'])
    return df

def get_abnormalities(diag,diag_save_path=None):
    if isinstance(diag,str):
        diag=load(diag)
    diagnoses=[]
    errors=0
    for item in diag['Abnormalities'].str.replace('\n','').to_list():
        try:
            x=ast.literal_eval(item)
        except:
            errors+=1
            continue
        if type(x)==list and len(x)>0:
            for item in x:
                try:
                    diagnoses.append(item['abnormality'])
                except:
                    errors+=1

    diagnoses=list(set(diagnoses))

    print(f'Errors: {errors}')
    print(f'Number of abnormal findings: {len(diagnoses)}')

    if diag_save_path is not None:
        df = pd.DataFrame(diagnoses, columns=["Diagnoses"])
        df.to_csv(diag_save_path, index=False)

    return diagnoses


get_diagnoses_0 = """
Below is a large list of radiological findings and diagnoses. Many items in this list may be synonyms or near-synonyms referring to the same underlying concept. I would like you to produce a Python dictionary that groups these terms together. Specifically, follow these instructions:

- For each concept that has synonyms or closely related terms, choose one standard radiological term as the key.
- Under that key, list all variants, synonyms, or closely related terms from the provided list as the dictionary value (a list of strings).
- If a term does not have any synonyms, it should appear as a key with a single-value list containing just that term.
- Focus on grouping terms that radiologists would consider essentially synonymous or describing the same imaging finding.
- Preserve all unique terms from the original list so that every term appears in the final dictionary, either as a single-item list or grouped under one of the synonyms.
- The values of the dictionary should include all terms from the original list, and no term should be repeated in multiple groups.
- Unless necessary, the organ name should not be included in the key term. For example, use "cyst" instead of "renal cyst" or "liver cyst".
- For each key in the dictionary, the corresponding value should contain all synonyms for that key, **including the key itself**.

Here is the list of findings/diagnoses to process:

%(diagnoses)s

Please provide the dictionary as Python code, using a dictionary literal with keys as strings and values as lists of strings. Call the dictionary "synonyms". Remember that ALL findings above must appear in the dictionary.
"""

get_diagnoses = """
Below is a large list of radiological findings and diagnoses. Many items in this list may be synonyms or near-synonyms referring to the same underlying concept. I would like you to produce a Python dictionary that groups these terms together. Specifically, follow these instructions:


- For each concept that has synonyms or closely related terms, choose one standard radiological term as the key.
- Under that key, list all variants, synonyms, or closely related terms from the provided list as the dictionary value (a list of strings).
- If a term does not have any synonyms, it should appear as a key with a single-value list containing just that term.
- Focus on grouping terms that radiologists would consider essentially synonymous or describing the same imaging finding.
- Preserve all unique terms from the original list so that every term appears in the final dictionary, either as a single-item list or grouped under one of the synonyms.
- The values of the dictionary should include all terms from the original list, and no term should be repeated in multiple groups.
- Unless necessary, the organ name should not be included in the key term. For example, use "cyst" instead of "renal cyst" or "liver cyst".
- For each key in the dictionary, the corresponding value should contain all synonyms for that key, **including the key itself**.

Here is the list of findings/diagnoses to process:

%(diagnoses)s

Please try to get keys for your dictionary from the finsings below. In case some findings are not synonyms with any of the findings below, you can create new keys for them. Remember that ALL findings above must appear in the dictionary.

%(synonyms)s
"""
def merge_dicts(dict1, dict2):
    """
    Merges two dictionaries. For shared keys, combines values and removes duplicates.

    Args:
        dict1 (dict): First dictionary.
        dict2 (dict): Second dictionary.

    Returns:
        dict: Merged dictionary with combined and deduplicated values for shared keys.
    """
    merged_dict = {}

    # Get all keys from both dictionaries
    all_keys = set(dict1.keys()).union(set(dict2.keys()))

    for key in all_keys:
        # Fetch values from both dictionaries; default to empty list
        values1 = dict1.get(key, [])
        values2 = dict2.get(key, [])
        
        # Ensure values are lists for easy merging
        if not isinstance(values1, list):
            values1 = [values1]
        if not isinstance(values2, list):
            values2 = [values2]
        
        # Combine values and remove duplicates
        merged_dict[key] = list(set(values1 + values2))

    return merged_dict

def summarize_diagnoses(diagnoses,base_url='http://0.0.0.0:8000/v1',batch=100,save_name=None):
    if isinstance(diagnoses,str):
        diagnoses=get_abnormalities(diagnoses)
    start=0
    end=batch
    non_added_values=[]
    while True:
        print('Start:',start)
        if end>len(diagnoses):
            end=len(diagnoses)
        d=diagnoses[start:end]+non_added_values

        keys=''
        for item in d:
            keys+=item+', '

        if start==0:
            prompt=get_diagnoses_0 % {'diagnoses':str(d)}
        else:
            prompt=get_diagnoses % {'diagnoses':str(d),'synonyms':keys}
        message= [{"role": "system", "content": system+' \n '+observations},
                  {"role": "user", "content": prompt}]
        
        if start!=0:
            old_syn=copy.deepcopy(synonyms)

        conver,answer=SendMessageAPI(text=None, conver=message, base_url=base_url)
            
        new_syns=interpret_output(answer,step='synonyms')
        try:
            new_syns=ast.literal_eval(new_syns)
        except:
            prompt="""You returned a non-valid python dictionary, I had an error using ast.literal_eval() for it. Please provide a valid python dictionary named synonyms. Answer with just the disctionary."""
            conver.append({"role": "user", "content": prompt})
            conver,answer=SendMessageAPI(text=None, conver=conver, base_url=base_url)
            new_syns=interpret_output(answer,step='synonyms')
            new_syns=ast.literal_eval(new_syns)

        if start!=0:
            synonyms=merge_dicts(synonyms, new_syns)
        else:
            synonyms=new_syns

        if start!=0:

            #check if all values are in synonyms
            new_values = list(chain.from_iterable(synonyms.values()))
            non_added_values=[]
            for value in d:
                if value not in new_values:
                    non_added_values.append(value)

            if len(non_added_values)>0:
                prompt="""The synonyms dictionary you provide is incomplete. Please add to it the itens below and send me the updated dictionary (the entire dictionary, name it synonyms). 
                        If the synonyms below are not synonyms with any of the itens in the dictionary, you can create new keys for them. Otherwise, add them to the existing keys. \n"""
                prompt+='Add these findings to values:'
                prompt+=str(non_added_values)
                print('Missed values:',non_added_values)
                conver.append({"role": "user", "content": prompt})

                _,answer=SendMessageAPI(text=None, conver=conver, base_url=base_url)
                print('Answer:',answer)
                new_syns=interpret_output(answer,step='synonyms')
                try:
                    new_syns=ast.literal_eval(new_syns)
                except:
                    prompt="""You returned a non-valid python dictionary, I had an error using ast.literal_eval() for it. Please provide a valid python dictionary named synonyms. Answer with just the disctionary."""
                    conver.append({"role": "user", "content": prompt})
                    conver,answer=SendMessageAPI(text=None, conver=conver, base_url=base_url)
                    new_syns=interpret_output(answer,step='synonyms')
                    new_syns=ast.literal_eval(new_syns)
                synonyms=merge_dicts(synonyms, new_syns)

                new_values = list(chain.from_iterable(synonyms.values()))
                non_added_values=[]
                for value in d:
                    if value not in new_values:
                        non_added_values.append(value)
                if len(non_added_values)>0:
                    print('Still not added values:',non_added_values)

        #check if all values previously in synonyms are still there


        print('# of synonym groups:',len(synonyms))
        print('# of words in synonyms:',len(list(chain.from_iterable(synonyms.values()))))

        if save_name is not None:
            #remove file if it exists
            if start==0 and os.path.exists(save_name):
                os.remove(save_name)
            with open(save_name, 'w') as file:
                file.write(str(synonyms))

        start+=batch
        end+=batch
        if start>=len(diagnoses):
            break

    return synonyms

def get_standard_key(finding, synonyms_dict,sub_organ=None):
    """
    Given a finding (string) and the synonyms_dict (a dictionary where
    keys are 'standard' terms and values are lists of synonym strings),
    this function returns the key whose value list contains the given finding,
    ignoring case. If no match is found, it returns None.
    """
    finding_lower = finding.lower()
    if sub_organ is not None:
        sub_organ=sub_organ.lower()
    for key, synonym_list in synonyms_dict.items():
        if sub_organ is not None:
            if any(sub_organ == synonym.lower() for synonym in synonym_list):
                return key
        # Check if the lowercase version of finding matches any lowercase synonym
        if any(finding_lower == synonym.lower() for synonym in synonym_list):
            return key
    return None

def count_findings(diagnoses_csv,synonyms_dict,organ='all'):
    import ast
    # Load the CSV file
    diag = load(diagnoses_csv)
    diagnoses={}
    LLM_errors=0
    report=0
    missing_from_synonym_dict=[]
    for item in diag['Abnormalities'].str.replace('\n','').to_list():
        try:
            x=ast.literal_eval(item)
        except:
            LLM_errors+=1
            #print('Error here')
            continue
        general_diag=[]
        if type(x)==list and len(x)>0:
            for item in x:
                if organ!='all':
                    if 'organ' not in item:
                        continue
                    if item['organ'] not in organ:
                        continue
                try:
                    y=item['abnormality']
                except:
                    LLM_errors+=1
                    continue
                if synonyms_dict is not None:
                    d=get_standard_key(y, synonyms_dict)
                else:
                    d=y
                if d is None:
                    missing_from_synonym_dict.append(y)
                    continue
                general_diag.append(d)
                
            general_diag=list(set(general_diag))
            #print(general_diag)
            for d in general_diag:
                if d not in diagnoses:
                    diagnoses[d]=1
                else:
                    diagnoses[d]+=1
            report+=1

    print('LLM errors:',LLM_errors)
    missing_from_synonym_dict=list(set(missing_from_synonym_dict))
    print('Diagnoses:',len(diagnoses))
    print('Missing from synonym dict:',len(missing_from_synonym_dict))
    print('Reports:',report)

    return diagnoses,missing_from_synonym_dict


def plot_top_diseases(results_LLM, N=10, minimum=1, flip_axes=False,organ='all',synonyms_dict=None,font=10):
    """
    Plots a bar chart of the top N diseases by occurrence.
    The largest-occurrence disease will be at the top and bars extend from left to right,
    unless flip_axes is True, in which case the occurrences are plotted on the y-axis.
    """
    disease_dict,_=count_findings(results_LLM,synonyms_dict,organ=organ)
    print('Disease dict:',len(disease_dict))
    # Sort diseases by occurrences in descending order
    sorted_items = sorted(disease_dict.items(), key=lambda x: x[1], reverse=True)

    # Select top N and filter by minimum threshold
    top_items = [item for item in sorted_items[:N] if item[1] >= minimum]

    # Unpack into lists for plotting
    diseases, occurrences = zip(*top_items)

    # Adjust figure size dynamically
    long = max(6, len(top_items) * 0.2)
    short = 6
    if flip_axes:
        plt.figure(figsize=(long, short))
    else:
        plt.figure(figsize=(short, long))

    if flip_axes:
        # Vertical bar plot (occurrences on y-axis)
        plt.bar(diseases, occurrences, color='skyblue')
        plt.ylabel('Occurrences',fontsize=font)
        plt.xlabel('Diseases',fontsize=font)
        title=f'Top {str(N)} Diseases by Occurrence'
        if organ!='all':
            title+=' for '+organ[0].capitalize()
        plt.title(title,fontsize=font)
        plt.xticks(rotation=90, ha="center")  # Rotate x-axis labels for better readability
        plt.gca().margins(x=0.01, y=0.1)  # Reduce internal padding
    else:
        # Horizontal bar plot (default behavior)
        plt.barh(diseases, occurrences, color='skyblue')
        plt.xlabel('Occurrences')
        plt.title(f'Top {N} Diseases by Occurrence',fontsize=font)
        plt.gca().invert_yaxis()  # Largest at the top
        plt.gca().margins(y=0.01, x=0.1)  # Reduce internal padding

    # Apply minimal padding
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
    plt.xticks(fontsize=font)  # X-axis ticks font size
    plt.yticks(fontsize=font)  # Y-axis ticks font size
    
    # Show the plot
    plt.show()

possible_cancers= [
    "tumor",
    "mass",
    "metastasis",
    "metastases",
    "carcinomatosis",
    "hepatocellular carcinoma",
    "tumor thrombus",
    "lesion",
    "lesions",
    "nodule",
    "nodules",
    "cystic lesion",
    "sclerotic lesion",
    "sclerotic lesions",
    "lytic lesion",
    "nodular contour",
    "nodular density",
    "nodular opacity",
    "ground-glass nodule",
    "soft tissue mass",
    "cancer",
    "neoplasm",
    'hyperdensity',
    'hypodensity'
]


organ_dict = {
    "liver": ["liver"],
    "kidney": ["kidney", "left kidney","nephrectomy bed"],
    "pancreas": ["pancreas", "pancreatic head"],
    "adrenal gland": ["adrenal gland"],
    "lung": ["lung"],
    "reproductive organs": ["reproductive organs"],
    "vagina": ["vagina"],
    "uterus": ["uterus"],
    "prostate": ["prostate"],
    "spleen": ["spleen"],
    "gi tract": ["gi tract","gastric remnant"],
    "stomach": ["stomach"],
    "duodenum": ["duodenum"],
    "small bowel": ["small bowel", "small bowel mesentery"],
    "rectum": ["rectum"],
    "peritoneum": ["peritoneum", "peritoneum/retroperitoneum"],
    "bone": ["bone", "pubic bone", "right iliac bone", "vertebral body", "skeleton", "sacrum", "acetabulum"],
    "soft tissue": ["soft tissue", "subcutaneous tissue", "retroperitoneal soft tissue",
                    "extraperitoneal soft tissue", "subcutaneous soft tissue"],
    "retroperitoneum": ["retroperitoneum", "left retroperitoneum", "retroperitoneal", "retroperitoneal space"],
    "ovary": ["ovary", "ovarie"],
    "bladder": ["bladder"],
    "gallbladder": ["gallbladder"],
    "pelvis": ["pelvis", "right pelvi"],
    "mesentery": ["mesentery", "mesenteric", "mesentery or omentum", "small bowel mesentery"],
    "lymphatic system": ["lymph node", "lymphatic system"],
    "bile duct": ["bile duct", "common bile duct"],
    "vasculature": ["vasculature", "vein", "artery"],
    "abdomen": ["abdomen", "anterior abdominal wall", "hemidiaphragm"],
    "skin": ["skin"],
    "omentum": ["omentum"],
    "pericardium": ["pericardium"],
    "breast": ["breast"],
    "bartholin's gland": ["bartholin's gland"],
    "musculature": ["musculature", "left iliopsoas muscle"],
    "presacral": ["presacral"],
    "endometrium": ["endometrium"],
    "diaphragm": ["diaphragm"],
    "colon": ["colon"],
    "esophagus": ["esophagus"],
}

def count_organs(diagnoses_csv,synonyms_dict,diseases=['cancer','lesion','tumor','hypodensities','hyperdensities'],organ_dict=organ_dict):
    import ast
    # Load the CSV file
    diag = load(diagnoses_csv)
    organ_counts={}
    LLM_errors=0
    report=0
    missing_from_synonym_dict=[]
    for item in diag['Abnormalities'].str.replace('\n','').to_list():
        try:
            x=ast.literal_eval(item)
        except:
            LLM_errors+=1
            #print('Error here')
            continue
        #print(x)
        if type(x)==list and len(x)>0:
            organs_tumor=[]
            for item in x:
                #print(item)
                if diseases!='all':
                    try:
                        disease=item['abnormality']
                        if synonyms_dict is not None:
                            disease=get_standard_key(disease, synonyms_dict)
                        #print(disease)
                        if disease not in diseases:
                            continue
                    except:
                        continue
                try:
                    y=item['organ'].lower()
                    y=get_standard_key(y, organ_dict)
                    if y[-1]=='s' and y not in ['pancreas','uterus','reproductive organs','pelvis']:
                        y=y[:-1]
                except:
                    LLM_errors+=1
                    continue
                organs_tumor.append(y)
            organs_tumor=list(set(organs_tumor))
            for y in organs_tumor:
                if y not in organ_counts:
                    organ_counts[y]=1
                else:
                    organ_counts[y]+=1
            report+=1
                

    print('LLM errors:',LLM_errors)
    print('Reports:',report)

    return organ_counts

def plot_cancer_organs(results_LLM, N=10, minimum=1, flip_axes=False, organ='all', synonyms_dict=None,
                       diseases=possible_cancers, font=20, log_scale=False):
    """
    Plots a bar chart of the top N diseases by occurrence.
    The largest-occurrence disease will be at the top and bars extend from left to right,
    unless flip_axes is True, in which case the occurrences are plotted on the y-axis.
    """
    disease_dict = count_organs(results_LLM, synonyms_dict, diseases=diseases)
    print('Disease dict:', disease_dict)
    print('Disease dict:', len(disease_dict))
    
    # Sort diseases by occurrences in descending order
    sorted_items = sorted(disease_dict.items(), key=lambda x: x[1], reverse=True)

    # Select top N and filter by minimum threshold
    top_items = [item for item in sorted_items[:N] if item[1] >= minimum]

    # Unpack into lists for plotting
    diseases, occurrences = zip(*top_items)

    # Adjust figure size dynamically
    long = max(6, len(top_items) * 0.2)
    short = 6
    if flip_axes:
        plt.figure(figsize=(long, short))
    else:
        plt.figure(figsize=(short, long))

    if flip_axes:
        # Vertical bar plot (occurrences on y-axis)
        plt.bar(diseases, occurrences, color='skyblue', log=log_scale)
        plt.ylabel('Occurrences', fontsize=font)
        plt.xlabel('Organs', fontsize=font)
        title = f'Number of tumor reports per organ'
        plt.title(title)
        plt.xticks(rotation=90, ha="center")  # Rotate x-axis labels for better readability
        plt.gca().margins(x=0.01, y=0.1)  # Reduce internal padding
    else:
        # Horizontal bar plot (default behavior)
        plt.barh(diseases, occurrences, color='skyblue', log=log_scale)
        plt.xlabel('Occurrences', fontsize=font)
        plt.title(f'Number of tumor reports per organ', fontsize=font)
        plt.gca().invert_yaxis()  # Largest at the top
        plt.gca().margins(y=0.01, x=0.1)  # Reduce internal padding

    # Apply minimal padding
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.15)
    
    # Tick Labels
    plt.xticks(fontsize=font)  # X-axis ticks font size
    plt.yticks(fontsize=font)  # Y-axis ticks font size

    # Show the plot
    plt.show()



def select_disease_organ(data, diseases, organs, organ_dict=organ_dict, synonyms_dict=None):
    llm_out = load(data)
    header = llm_out.columns.to_list()

    LLM_errors = 0
    cases = {}

    rows_to_add = []

    # Use itertuples with name=None to get plain tuples
    for tup in tqdm.tqdm(llm_out.itertuples(index=False, name=None), total=len(llm_out)):
        # Create a dict mapping header -> value directly from the tuple
        row_dict = dict(zip(header, tup))
        
        abnormalities = row_dict.get('Abnormalities')
        if not isinstance(abnormalities, str):
            LLM_errors += 1
            continue

        item = abnormalities.replace('\n', '')

        try:
            x = ast.literal_eval(item)
        except:
            LLM_errors += 1
            continue

        add = False
        row_added = False
        organs_added = []

        try:
            if isinstance(x, list) and len(x) > 0:
                for sub_item in x:
                    organ = sub_item.get('organ', '')
                    sub_organ=sub_item.get('location_inside_organ', '')
                    description = sub_item.get('description', '')
                    if "unremarkable" in description.lower() or "post-operative" in description.lower() or "absen" in description.lower() or "enlarged" in description.lower():
                        continue
                    organ = get_standard_key(organ, organ_dict,sub_organ)
                    diag = sub_item.get('abnormality', '')
                    if synonyms_dict is not None:
                        diag = get_standard_key(diag, synonyms_dict)

                    if diag in diseases and organ not in organs_added:
                        add = True
                        if organ not in cases:
                            cases[organ] = []
                        cases[organ].append(row_dict)
                        organs_added.append(organ)
        except:
            LLM_errors += 1
            continue

        if add:
            rows_to_add.append(row_dict)

    df = pd.DataFrame(rows_to_add, columns=header)
    return df, cases