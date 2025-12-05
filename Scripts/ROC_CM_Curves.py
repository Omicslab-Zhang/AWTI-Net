###########################################  1 step: import packages  ##################################
### The package from the python system ###
import os
import time
import torch
import glob
import cv2
import torchvision
import warnings
import scipy.ndimage
import torch.nn.functional as F
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt
import os.path as osp
import pandas as pd
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix,roc_auc_score
from sklearn.metrics import roc_curve, auc

# 绘制ROC曲线
def get_ROC_Train(y_test, y_prob, auc_value,result_path):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.4f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('Train-ROC-Train', fontsize=20)
    plt.legend(loc="lower right", fontsize=10)
    plt.savefig(result_path+"/Trans-ROC-Train.png")
    plt.show()

########################  绘制混淆矩阵  #####################
def get_CM_Train(y_test, y_pred,result_path):
    sns.set(style='white')
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(np.eye(2), annot=cm, fmt='g', annot_kws={'size': 30},
            cmap=sns.color_palette(['Purple', 'Wheat'], as_cmap=True), cbar=True,
            yticklabels=['Non-functional', 'Functional'], xticklabels=['Non-functional', 'Functional'], ax=ax)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('bottom')
    ax.tick_params(labelsize=20, length=0)
    ax.set_title('Confusion Matrix-Train', size=24, pad=20)
    ax.set_xlabel('Predicted Values', size=20)
    ax.set_ylabel('Actual Values', size=20)
    additional_texts = ['(True Positive)', '(False Negative)', '(False Positive)', '(True Negative)']
    for text_elt, additional_text in zip(ax.texts, additional_texts):
        ax.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
            ha='center', va='top', size=15)
    plt.tight_layout()
    plt.savefig(result_path+"/Trans-CM-Train.png")
    plt.show()

# 绘制ROC曲线
def get_ROC_Valid(y_test, y_prob, auc_value,result_path):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label='ROC curve (area = %0.4f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', lw=2.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('Valid-ROC-Valid', fontsize=20)
    plt.legend(loc="lower right", fontsize=15)
    plt.savefig(result_path+"/Trans-ROC-Valid.png")
    plt.show()

########################  绘制混淆矩阵  #####################
def get_CM_Valid(y_test, y_pred,result_path):
    sns.set(style='white')
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(np.eye(2), annot=cm, fmt='g', annot_kws={'size': 30},
            cmap=sns.color_palette(['Purple', 'Wheat'], as_cmap=True), cbar=True,
            yticklabels=['Non-functional', 'Functional'], xticklabels=['Non-functional', 'Functional'], ax=ax)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('bottom')
    ax.tick_params(labelsize=20, length=0)
    ax.set_title('Confusion Matrix-Valid', size=24, pad=20)
    ax.set_xlabel('Predicted Values', size=20)
    ax.set_ylabel('True Values', size=20)
    additional_texts = ['(True Positive)', '(False Negative)', '(False Positive)', '(True Negative)']
    for text_elt, additional_text in zip(ax.texts, additional_texts):
        ax.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
            ha='center', va='top', size=15)
    plt.tight_layout()
    plt.savefig(result_path+"/Trans-CM-Valid.png")
    plt.show()


# 绘制ROC曲线
def get_ROC_Test193(y_test, y_prob, auc_value,result_path):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=2.5, label='ROC curve (area = %0.4f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', lw=2.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('Test-ROC-Test', fontsize=20)
    plt.legend(loc="lower right", fontsize=15)
    plt.savefig(result_path+"/Trans-ROC-Test193.png")
    plt.show()

########################  绘制混淆矩阵  #####################
def get_CM_Test193(y_test, y_pred,result_path):
    sns.set(style='white')
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(np.eye(2), annot=cm, fmt='g', annot_kws={'size': 30},
            cmap=sns.color_palette(['Purple', 'Wheat'], as_cmap=True), cbar=True,
            yticklabels=['Non-functional', 'Functional'], xticklabels=['Non-functional', 'Functional'], ax=ax)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('bottom')
    ax.tick_params(labelsize=20, length=0)
    ax.set_title('Confusion Matrix-Test', size=24, pad=20)
    ax.set_xlabel('Predicted Values', size=20)
    ax.set_ylabel('True Values', size=20)
    additional_texts = ['(True Positive)', '(False Negative)', '(False Positive)', '(True Negative)']
    for text_elt, additional_text in zip(ax.texts, additional_texts):
        ax.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
            ha='center', va='top', size=15)
    plt.tight_layout()
    plt.savefig(result_path+"/Trans-CM-Test193.png")
    plt.show()

# 绘制ROC曲线
def get_ROC_Test30(y_test, y_prob, auc_value,result_path):
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, color='red', lw=2.5, label='ROC curve (area = %0.4f)' % auc_value)
    plt.plot([0, 1], [0, 1], color='navy', lw=2.5, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=20)
    plt.ylabel('True Positive Rate', fontsize=20)
    plt.title('Test-ROC-Test', fontsize=20)
    plt.legend(loc="lower right", fontsize=15)
    plt.savefig(result_path+"/Trans-ROC-Test30.png")
    plt.show()

########################  绘制混淆矩阵  #####################
def get_CM_Test30(y_test, y_pred,result_path):
    sns.set(style='white')
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(np.eye(2), annot=cm, fmt='g', annot_kws={'size': 30},
            cmap=sns.color_palette(['Purple', 'Wheat'], as_cmap=True), cbar=True,
            yticklabels=['Non-functional', 'Functional'], xticklabels=['Non-functional', 'Functional'], ax=ax)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position('bottom')
    ax.tick_params(labelsize=20, length=0)
    ax.set_title('Confusion Matrix-Test', size=24, pad=20)
    ax.set_xlabel('Predicted Values', size=20)
    ax.set_ylabel('True Values', size=20)
    additional_texts = ['(True Positive)', '(False Negative)', '(False Positive)', '(True Negative)']
    for text_elt, additional_text in zip(ax.texts, additional_texts):
        ax.text(*text_elt.get_position(), '\n' + additional_text, color=text_elt.get_color(),
            ha='center', va='top', size=15)
    plt.tight_layout()
    plt.savefig(result_path+"/Trans-CM-Test30.png")
    plt.show()