# Databricks notebook source
import PyPDF2
from pathlib import Path
import sys
import os
import json


def validate_path(doc_path, allowed_dir):
    """
    Validate the document path to ensure it is within the allowed directory.

    Args:
        doc_path (str): The path to the document.
        allowed_dir (str): The allowed directory path.
    """
    path = Path(doc_path).resolve()
    if not str(path).startswith(str(allowed_dir)):
        raise ValueError(f"Document path {doc_path} is not within the allowed directory {allowed_dir}.")
    return path

def extract_text_from_pdf(pdf_path, allowed_dir):
    """
    Extract text from a PDF file.

    Args:
        pdf_path (str): The path to the PDF file.
    """
    path = validate_path(pdf_path, allowed_dir)
    with open(path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page in reader.pages:
            text += page.extract_text() + '\n'

    return text.strip()

def parse_json_file(json_response):
    """
    Parse a JSON file and return its content.

    Args:
        json_response (str): The JSON response as a string.
    """
    result_str = str(json_response)
    if "```" in result_str:
        result_str = result_str.split("```")[1].split("```")[0].strip()

    return result_str