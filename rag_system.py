#!/usr/bin/env python3
"""
RAG System - LlamaIndex & LangChain Integration
结合LlamaIndex和LangChain的完整RAG（检索增强生成）系统

Author: joytianya
Date: 2025-06-06
License: MIT

This module provides a comprehensive RAG system that combines the strengths
of both LlamaIndex and LangChain for document processing and question answering.
"""

import os
import sys
import json
import logging
import asyncio
from typing import List, Optional, Dict, Any, Union
from pathlib import Path
from datetime import datetime
import warnings

# Core dependencies
try:
    from llama_index.core import (
        VectorStoreIndex, 
        SimpleDirectoryReader, 
        ServiceContext,
        StorageContext,
        Settings,
        Document
  )
    from llama_index.core.node_parser import SimpleNodeParser
    from llama_index.core.memory import ChatMemoryBuffer
    from llama_index.core.chat_engine import CondenseQuestionChatEngine
