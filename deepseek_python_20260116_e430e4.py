#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# 🚀 ULTRA FILE SEARCH BOT - يدعم حتى 2GB - أسرع وأقوى

import os
import sys
import json
import csv
import sqlite3
import pandas as pd
import numpy as np
import re
import zipfile
import hashlib
import chardet
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Generator

# Telegram
from telegram import (
    Update, 
    InlineKeyboardButton, 
    InlineKeyboardMarkup,
    InputFile,
    InputMediaDocument,
    ReplyKeyboardMarkup,
    KeyboardButton
)
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    CallbackQueryHandler,
    ContextTypes,
    ConversationHandler,
    PicklePersistence
)

# ==================== CONFIGURATION ====================
class Config:
    # Telegram Bot Token
    BOT_TOKEN = "7611903521:AAFv1xiXkFlJMErbpk7aTpKMS79bcnPPNSU"  # ✅ تم التعديل
    
    # Admin IDs
    ADMIN_IDS = [8493388920]
    
    # File size limits (2GB = 2 * 1024 * 1024 * 1024)
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
    MAX_MEMORY_USAGE = 1 * 1024 * 1024 * 1024  # 1GB RAM
    CHUNK_SIZE = 10 * 1024 * 1024  # 10MB chunks
    
    # Database
    DB_PATH = "ultra_file_search.db"
    CACHE_DIR = Path("cache")
    TEMP_DIR = Path("temp_files")
    LOG_DIR = Path("logs")
    
    # Performance
    MAX_WORKERS = 4  # تقليل العدد لتجنب multiprocessing
    CACHE_TTL = 3600  # 1 hour
    BATCH_SIZE = 1000
    TIMEOUT_SECONDS = 300  # 5 minutes
    
    # Security
    ALLOWED_EXTENSIONS = {
        '.csv', '.json', '.txt', '.xlsx', '.xls', '.xlsm',
        '.db', '.sqlite', '.sqlite3', '.parquet', '.feather',
        '.tsv', '.xml', '.yaml', '.yml'
    }
    
    MAX_FILES_PER_USER = 10
    MAX_SEARCHES_PER_DAY = 100
    
    @classmethod
    def setup(cls):
        """Create necessary directories"""
        for directory in [cls.CACHE_DIR, cls.TEMP_DIR, cls.LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

# ==================== ENHANCED LOGGING ====================
import logging

class EnhancedLogger:
    @staticmethod
    def setup():
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)
        
        # Console handler
        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console.setFormatter(console_format)
        
        logger.addHandler(console)
        
        return logger

logger = EnhancedLogger.setup()

# ==================== SIMPLE FILE MANAGER ====================
class SimpleFileManager:
    """مدير ملفات مبسط بدون مكتبات محظورة"""
    
    def __init__(self):
        self.active_files = {}
        self.file_cache = {}
    
    def detect_encoding(self, file_path: str) -> str:
        """كشف ترميز الملف"""
        try:
            with open(file_path, 'rb') as f:
                raw_data = f.read(10000)
                
                # Simple encoding detection
                try:
                    raw_data.decode('utf-8')
                    return 'utf-8'
                except:
                    try:
                        raw_data.decode('utf-8-sig')
                        return 'utf-8-sig'
                    except:
                        try:
                            raw_data.decode('cp1256')
                            return 'cp1256'
                        except:
                            return 'latin-1'
        except Exception as e:
            logger.error(f"Encoding detection error: {e}")
            return 'utf-8'
    
    def get_file_size(self, file_path: str) -> int:
        """الحصول على حجم الملف"""
        try:
            return os.path.getsize(file_path)
        except:
            return 0
    
    def load_csv(self, file_path: str) -> Dict:
        """تحميل ملف CSV"""
        try:
            encoding = self.detect_encoding(file_path)
            file_size = self.get_file_size(file_path)
            
            # Load based on size
            if file_size < 50 * 1024 * 1024:  # أقل من 50MB
                df = pd.read_csv(file_path, encoding=encoding, low_memory=False)
                loaded_fully = True
            else:
                # تحميل أول 10000 سطر للملفات الكبيرة
                df = pd.read_csv(file_path, encoding=encoding, nrows=10000, low_memory=False)
                loaded_fully = False
            
            return {
                'type': 'csv',
                'data': df,
                'size': file_size,
                'rows': len(df),
                'columns': list(df.columns),
                'loaded_fully': loaded_fully
            }
                
        except Exception as e:
            logger.error(f"CSV loading error: {e}")
            return {'error': str(e)}
    
    def load_json(self, file_path: str) -> Dict:
        """تحميل ملف JSON"""
        try:
            file_size = self.get_file_size(file_path)
            
            if file_size < 10 * 1024 * 1024:  # أقل من 10MB
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                if isinstance(data, list):
                    df = pd.DataFrame(data[:10000])  # Limit to 10k records
                elif isinstance(data, dict):
                    df = pd.DataFrame([data])
                else:
                    df = pd.DataFrame({'data': [str(data)]})
                
                return {
                    'type': 'json',
                    'data': df,
                    'size': file_size,
                    'rows': len(df),
                    'columns': list(df.columns)
                }
            else:
                # For large JSON, read line by line
                data = []
                with open(file_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 10000:  # Limit to 10k lines
                            break
                        try:
                            item = json.loads(line.strip())
                            data.append(item)
                        except:
                            continue
                
                df = pd.DataFrame(data) if data else pd.DataFrame()
                
                return {
                    'type': 'json',
                    'data': df,
                    'size': file_size,
                    'rows': len(df),
                    'columns': list(df.columns) if not df.empty else [],
                    'sample_only': True
                }
                
        except Exception as e:
            logger.error(f"JSON loading error: {e}")
            return {'error': str(e)}
    
    def load_excel(self, file_path: str) -> Dict:
        """تحميل ملف Excel"""
        try:
            # Read first sheet
            df = pd.read_excel(file_path, nrows=10000)  # Limit to 10k rows
            
            return {
                'type': 'excel',
                'data': df,
                'size': self.get_file_size(file_path),
                'rows': len(df),
                'columns': list(df.columns)
            }
            
        except Exception as e:
            logger.error(f"Excel loading error: {e}")
            return {'error': str(e)}
    
    def load_sqlite(self, file_path: str) -> Dict:
        """تحميل قاعدة بيانات SQLite"""
        try:
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            
            # Get all tables
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if not tables:
                conn.close()
                return {'error': 'No tables found'}
            
            # Get first table data
            table = tables[0]
            df = pd.read_sql_query(f"SELECT * FROM {table} LIMIT 10000", conn)
            
            conn.close()
            
            return {
                'type': 'sqlite',
                'data': df,
                'size': self.get_file_size(file_path),
                'rows': len(df),
                'columns': list(df.columns),
                'table': table
            }
            
        except Exception as e:
            logger.error(f"SQLite loading error: {e}")
            return {'error': str(e)}
    
    def load_text(self, file_path: str) -> Dict:
        """تحميل ملف نصي"""
        try:
            encoding = self.detect_encoding(file_path)
            file_size = self.get_file_size(file_path)
            
            data = []
            with open(file_path, 'r', encoding=encoding) as f:
                for i, line in enumerate(f):
                    if i >= 10000:  # Limit to 10k lines
                        break
                    if line.strip():
                        data.append({
                            'line': i + 1,
                            'text': line.strip()[:500]  # Limit line length
                        })
            
            df = pd.DataFrame(data)
            
            return {
                'type': 'text',
                'data': df,
                'size': file_size,
                'rows': len(df)
            }
                
        except Exception as e:
            logger.error(f"Text loading error: {e}")
            return {'error': str(e)}
    
    def search_in_data(self, data: Dict, query: str) -> Dict:
        """بحث في البيانات"""
        results = {
            'exact_matches': [],
            'partial_matches': [],
            'column_stats': {},
            'search_time': None
        }
        
        start_time = time.time()
        
        try:
            df = data.get('data', pd.DataFrame())
            if df.empty:
                return results
            
            query_lower = query.lower()
            
            # Search in each text column
            for column in df.columns:
                if df[column].dtype == 'object':
                    col_results = {
                        'column': column,
                        'exact_rows': [],
                        'partial_rows': []
                    }
                    
                    # Exact match
                    exact_mask = df[column].astype(str).str.lower() == query_lower
                    if exact_mask.any():
                        exact_rows = df[exact_mask].head(10)
                        for _, row in exact_rows.iterrows():
                            col_results['exact_rows'].append(row.to_dict())
                    
                    # Partial match
                    partial_mask = df[column].astype(str).str.lower().str.contains(query_lower, na=False)
                    if partial_mask.any() and not exact_mask.all():
                        partial_rows = df[partial_mask & ~exact_mask].head(10)
                        for _, row in partial_rows.iterrows():
                            col_results['partial_rows'].append(row.to_dict())
                    
                    # Add to results
                    if col_results['exact_rows'] or col_results['partial_rows']:
                        results['exact_matches'].extend(col_results['exact_rows'])
                        results['partial_matches'].extend(col_results['partial_rows'])
                        
                        results['column_stats'][column] = {
                            'exact': len(col_results['exact_rows']),
                            'partial': len(col_results['partial_rows'])
                        }
            
            results['search_time'] = time.time() - start_time
            
            return results
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            results['error'] = str(e)
            return results
    
    def simple_search(self, file_path: str, query: str, file_type: str) -> Dict:
        """بحث مبسط"""
        try:
            # Load file based on type
            if file_type == 'csv':
                file_data = self.load_csv(file_path)
            elif file_type == 'json':
                file_data = self.load_json(file_path)
            elif file_type in ['xlsx', 'xls', 'xlsm']:
                file_data = self.load_excel(file_path)
            elif file_type in ['db', 'sqlite', 'sqlite3']:
                file_data = self.load_sqlite(file_path)
            else:
                file_data = self.load_text(file_path)
            
            if 'error' in file_data:
                return {'error': file_data['error']}
            
            # Search in loaded data
            return self.search_in_data(file_data, query)
            
        except Exception as e:
            logger.error(f"Simple search error: {e}")
            return {'error': str(e)}

# ==================== SIMPLE DATABASE ====================
class SimpleDatabase:
    """قاعدة بيانات مبسطة"""
    
    def __init__(self):
        self.db_path = Config.DB_PATH
        self.init_database()
    
    def init_database(self):
        """تهيئة قاعدة البيانات"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                username TEXT,
                first_name TEXT,
                last_name TEXT,
                join_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_active TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                total_files INTEGER DEFAULT 0,
                total_searches INTEGER DEFAULT 0
            )
        ''')
        
        # Files table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS files (
                file_id TEXT PRIMARY KEY,
                user_id INTEGER,
                original_name TEXT,
                file_size INTEGER,
                file_type TEXT,
                upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                file_hash TEXT
            )
        ''')
        
        # Searches table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS searches (
                search_id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                query TEXT,
                results_count INTEGER,
                search_time INTEGER,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def log_user_activity(self, user_id: int, username: str, first_name: str):
        """تسجيل نشاط المستخدم"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT OR REPLACE INTO users 
                (user_id, username, first_name, last_active) 
                VALUES (?, ?, ?, ?)
            ''', (user_id, username or '', first_name or '', datetime.now()))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"User activity logging error: {e}")
    
    def save_file_info(self, file_info: Dict):
        """حفظ معلومات الملف"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO files 
                (file_id, user_id, original_name, file_size, file_type, upload_date, file_hash)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                file_info.get('file_id'),
                file_info.get('user_id'),
                file_info.get('original_name'),
                file_info.get('file_size'),
                file_info.get('file_type'),
                datetime.now(),
                file_info.get('file_hash')
            ))
            
            # Update user stats
            cursor.execute('''
                UPDATE users 
                SET total_files = total_files + 1, 
                    last_active = ?
                WHERE user_id = ?
            ''', (datetime.now(), file_info.get('user_id')))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"File info saving error: {e}")
    
    def log_search(self, search_info: Dict):
        """تسجيل عملية البحث"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO searches 
                (user_id, query, results_count, search_time, timestamp)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                search_info.get('user_id'),
                search_info.get('query'),
                search_info.get('results_count', 0),
                search_info.get('search_time', 0),
                datetime.now()
            ))
            
            # Update user stats
            cursor.execute('''
                UPDATE users 
                SET total_searches = total_searches + 1,
                    last_active = ?
                WHERE user_id = ?
            ''', (datetime.now(), search_info.get('user_id')))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Search logging error: {e}")
    
    def get_user_stats(self, user_id: int) -> Dict:
        """الحصول على إحصائيات المستخدم"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT total_files, total_searches, join_date, last_active
                FROM users WHERE user_id = ?
            ''', (user_id,))
            
            row = cursor.fetchone()
            conn.close()
            
            if row:
                return {
                    'total_files': row[0],
                    'total_searches': row[1],
                    'join_date': row[2],
                    'last_active': row[3]
                }
            return {}
            
        except Exception as e:
            logger.error(f"Get user stats error: {e}")
            return {}
    
    def check_rate_limit(self, user_id: int) -> bool:
        """التحقق من حد الطلبات"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Count today's searches
            today = datetime.now().date()
            cursor.execute('''
                SELECT COUNT(*) FROM searches 
                WHERE user_id = ? AND DATE(timestamp) = ?
            ''', (user_id, today))
            
            count = cursor.fetchone()[0]
            conn.close()
            
            return count < Config.MAX_SEARCHES_PER_DAY
            
        except Exception as e:
            logger.error(f"Rate limit check error: {e}")
            return True

# ==================== POWERFUL TELEGRAM BOT ====================
class UltraFileSearchBot:
    def __init__(self, token: str):
        self.token = token
        self.file_manager = SimpleFileManager()
        self.database = SimpleDatabase()
        self.user_sessions = {}
        self.app = None
    
    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """بدء البوت"""
        user = update.effective_user
        
        # Log user activity
        self.database.log_user_activity(
            user.id, 
            user.username, 
            user.first_name
        )
        
        welcome = f"""
🚀 **أهلاً {user.first_name}!**

⚡ **Ultra File Search Bot v2.0**
بوت البحث في الملفات الأقوى!

📊 **المميزات:**
✅ دعم ملفات حتى 2GB
⚡ بحث سريع وفعال
🔍 دعم جميع أنواع الملفات
📈 إحصائيات مفصلة
💾 حفظ النتائج

📁 **الملفات المدعومة:**
• CSV, JSON, TXT
• Excel (XLSX, XLS)
• SQLite Databases
• XML, YAML

⚡ **كيفية الاستخدام:**
1. أرسل لي ملفاً
2. انتظر التحليل
3. اكتب ما تريد البحث عنه
4. احصل على النتائج

📊 **إحصائياتك:** /stats
🆘 **المساعدة:** /help

🚀 **أرسل ملفاً الآن لتبدأ!**
        """
        
        keyboard = [
            [KeyboardButton("📁 إرسال ملف")],
            [KeyboardButton("📊 إحصائياتي"), KeyboardButton("🆘 المساعدة")]
        ]
        
        reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
        
        await update.message.reply_text(
            welcome,
            parse_mode='Markdown',
            reply_markup=reply_markup
        )
    
    async def handle_document(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالجة الملفات المرسلة"""
        user = update.effective_user
        document = update.message.document
        
        if not document:
            await update.message.reply_text("❌ يرجى إرسال ملف صالح")
            return
        
        # Check file size
        file_size = document.file_size
        if file_size > Config.MAX_FILE_SIZE:
            await update.message.reply_text(
                f"❌ حجم الملف ({file_size / 1024 / 1024:.2f}MB) كبير جداً"
            )
            return
        
        # Check file extension
        file_name = document.file_name
        file_ext = os.path.splitext(file_name)[1].lower()
        
        if file_ext not in Config.ALLOWED_EXTENSIONS:
            await update.message.reply_text(
                f"❌ امتداد الملف غير مدعوم: {file_ext}"
            )
            return
        
        # Send processing message
        processing_msg = await update.message.reply_text(
            f"🔄 **جاري معالجة الملف:** `{file_name}`",
            parse_mode='Markdown'
        )
        
        try:
            # Download file
            file_dir = Config.TEMP_DIR / str(user.id)
            file_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = file_dir / f"{int(time.time())}_{file_name}"
            file = await context.bot.get_file(document.file_id)
            await file.download_to_drive(file_path)
            
            # Generate file hash
            file_hash = self.calculate_file_hash(file_path)
            
            # Process file
            start_time = time.time()
            
            # Load file based on type
            if file_ext == '.csv':
                file_data = self.file_manager.load_csv(file_path)
            elif file_ext == '.json':
                file_data = self.file_manager.load_json(file_path)
            elif file_ext in ['.xlsx', '.xls', '.xlsm']:
                file_data = self.file_manager.load_excel(file_path)
            elif file_ext in ['.db', '.sqlite', '.sqlite3']:
                file_data = self.file_manager.load_sqlite(file_path)
            else:
                file_data = self.file_manager.load_text(file_path)
            
            processing_time = time.time() - start_time
            
            if 'error' in file_data:
                await processing_msg.edit_text(
                    f"❌ **خطأ في معالجة الملف:**\n`{file_data['error']}`"
                )
                os.remove(file_path)
                return
            
            # Save to database
            self.database.save_file_info({
                'file_id': file_hash,
                'user_id': user.id,
                'original_name': file_name,
                'file_size': file_size,
                'file_type': file_ext.replace('.', ''),
                'file_hash': file_hash
            })
            
            # Prepare response
            rows_info = file_data['rows']
            columns = file_data.get('columns', [])
            
            await processing_msg.edit_text(
                f"✅ **تم معالجة الملف بنجاح!**\n\n"
                f"📊 **معلومات الملف:**\n"
                f"• **الصفوف:** {rows_info:,}\n"
                f"• **الأعمدة:** {len(columns)}\n"
                f"• **النوع:** {file_ext.upper()}\n\n"
                f"🔍 **اكتب الآن ما تريد البحث عنه:**",
                parse_mode='Markdown'
            )
            
            # Store session
            self.user_sessions[user.id] = {
                'file_path': str(file_path),
                'file_hash': file_hash,
                'file_name': file_name,
                'file_data': file_data,
                'file_size': file_size,
                'file_type': file_ext.replace('.', ''),
                'last_active': datetime.now()
            }
            
        except Exception as e:
            logger.error(f"File processing error: {e}")
            await processing_msg.edit_text(
                f"❌ **حدث خطأ:**\n`{str(e)[:100]}`"
            )
    
    async def handle_text(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """معالجة النصوص للبحث"""
        user = update.effective_user
        query = update.message.text.strip()
        
        if not query:
            await update.message.reply_text("❌ يرجى إدخال نص للبحث")
            return
        
        # Check if user has active file
        if user.id not in self.user_sessions:
            await update.message.reply_text(
                "❌ لم تقم بتحميل ملف بعد\n"
                "📁 يرجى إرسال ملف أولاً"
            )
            return
        
        # Check rate limit
        if not self.database.check_rate_limit(user.id):
            await update.message.reply_text(
                "⏰ **لقد تجاوزت الحد اليومي للبحث**"
            )
            return
        
        # Get session info
        session = self.user_sessions[user.id]
        file_path = session['file_path']
        file_data = session['file_data']
        
        # Send searching message
        search_msg = await update.message.reply_text(
            f"🔍 **جاري البحث عن:** `{query}`",
            parse_mode='Markdown'
        )
        
        try:
            start_time = time.time()
            
            # Perform search
            results = self.file_manager.search_in_data(file_data, query)
            search_time = time.time() - start_time
            
            if 'error' in results:
                await search_msg.edit_text(
                    f"❌ **خطأ في البحث:**\n`{results['error']}`"
                )
                return
            
            # Calculate total matches
            total_matches = len(results.get('exact_matches', [])) + len(results.get('partial_matches', []))
            
            # Log search
            self.database.log_search({
                'user_id': user.id,
                'query': query,
                'results_count': total_matches,
                'search_time': search_time
            })
            
            # Create report
            report = self.create_search_report(
                query, 
                results, 
                session, 
                search_time,
                total_matches
            )
            
            # Send results
            await search_msg.edit_text(
                report,
                parse_mode='Markdown'
            )
            
            # Send CSV if results found
            if total_matches > 0:
                await self.send_results_csv(update, results, session['file_name'])
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            await search_msg.edit_text(
                f"❌ **حدث خطأ:**\n`{str(e)[:100]}`"
            )
    
    def create_search_report(self, query: str, results: Dict, session: Dict, 
                           search_time: float, total_matches: int) -> str:
        """إنشاء تقرير عن نتائج البحث"""
        report = f"""
📊 **نتائج البحث عن:** `{query}`

✅ **الملخص:**
• إجمالي النتائج: **{total_matches:,}**
• وقت البحث: **{search_time:.2f} ثانية**
• الملف: **{session['file_name']}**
• الحجم: **{session['file_size'] / 1024 / 1024:.2f} MB**
"""
        
        # Add match types
        exact_count = len(results.get('exact_matches', []))
        partial_count = len(results.get('partial_matches', []))
        
        if exact_count > 0:
            report += f"\n✅ **التطابقات التامة:** {exact_count:,}"
        
        if partial_count > 0:
            report += f"\n🔍 **التطابقات الجزئية:** {partial_count:,}"
        
        # Add column statistics
        if results.get('column_stats'):
            report += "\n\n📈 **التوزيع على الأعمدة:**"
            for column, stats in list(results['column_stats'].items())[:3]:
                report += f"\n• `{column}`: {stats.get('exact', 0)} تام، {stats.get('partial', 0)} جزئي"
        
        # Add sample results
        sample_results = []
        if results.get('exact_matches'):
            sample_results.extend(results['exact_matches'][:2])
        elif results.get('partial_matches'):
            sample_results.extend(results['partial_matches'][:2])
        
        if sample_results:
            report += "\n\n🔎 **عينة من النتائج:**"
            for i, result in enumerate(sample_results[:2], 1):
                if isinstance(result, dict):
                    items = list(result.items())[:2]
                    result_text = "\n".join([f"  • {k}: {v}" for k, v in items])
                    report += f"\n\n**النتيجة {i}:**\n{result_text}"
        
        # Add recommendations
        if total_matches == 0:
            report += "\n\n💡 **لم يتم العثور على نتائج**\n"
            report += "• جرب البحث بكلمات مختلفة\n"
            report += "• تأكد من كتابة النص بشكل صحيح"
        
        report += f"\n\n⏰ **الوقت:** {datetime.now().strftime('%H:%M:%S')}"
        
        return report
    
    async def send_results_csv(self, update: Update, results: Dict, file_name: str):
        """إرسال النتائج كملف CSV"""
        try:
            # Combine all matches
            all_matches = []
            
            if results.get('exact_matches'):
                all_matches.extend(results['exact_matches'])
            
            if results.get('partial_matches'):
                all_matches.extend(results['partial_matches'])
            
            if not all_matches:
                return
            
            # Create CSV
            csv_data = []
            
            for i, match in enumerate(all_matches[:500], 1):  # Limit to 500 results
                if isinstance(match, dict):
                    row = {'result_number': i, **match}
                    csv_data.append(row)
            
            if csv_data:
                # Create DataFrame and save as CSV
                df = pd.DataFrame(csv_data)
                csv_path = Config.TEMP_DIR / f"results_{int(time.time())}.csv"
                df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                
                # Send file
                with open(csv_path, 'rb') as f:
                    await update.message.reply_document(
                        document=f,
                        filename=f"نتائج_{file_name}.csv",
                        caption="📥 **تم تصدير النتائج**"
                    )
                
                # Clean up
                os.remove(csv_path)
                
        except Exception as e:
            logger.error(f"CSV export error: {e}")
    
    async def stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """عرض إحصائيات المستخدم"""
        user = update.effective_user
        
        stats = self.database.get_user_stats(user.id)
        
        if stats:
            stats_msg = f"""
📊 **إحصائياتك الشخصية**

👤 **المعلومات:**
• الملفات المرفوعة: **{stats['total_files']}**
• عمليات البحث: **{stats['total_searches']}**
• آخر نشاط: **{stats['last_active']}**

💡 **نصائح:**
• الحد اليومي للبحث: {Config.MAX_SEARCHES_PER_DAY} عملية
• يمكنك البحث في نفس الملف عدة مرات
            """
        else:
            stats_msg = "📊 **ابدأ بإرسال ملف أولاً!**"
        
        await update.message.reply_text(stats_msg, parse_mode='Markdown')
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """أمر المساعدة"""
        help_text = """
🆘 **دليل الاستخدام**

🚀 **كيفية الاستخدام:**
1. أرسل `/start` لبدء البوت
2. أرسل ملفاً تريد البحث فيه
3. انتظر حتى يتم معالجة الملف
4. اكتب النص الذي تريد البحث عنه
5. احصل على النتائج

📁 **الملفات المدعومة:**
• CSV, JSON, TXT
• Excel (XLSX, XLS)
• SQLite Databases
• XML, YAML

🔍 **أنواع البحث:**
✅ **تطابق تام:** نفس النص تماماً
🔍 **تطابق جزئي:** يحتوي على النص

📊 **الأوامر:**
/start - بدء البوت
/stats - إحصائياتك
/help - المساعدة

⚠️ **ملاحظات:**
• الحد الأقصى للملف: 2GB
• الحد اليومي: 100 عملية بحث
• النتائج تحفظ كملف CSV
        """
        
        await update.message.reply_text(help_text, parse_mode='Markdown')
    
    async def clear_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """مسح الملفات المؤقتة"""
        user = update.effective_user
        
        try:
            # Clear user session
            if user.id in self.user_sessions:
                del self.user_sessions[user.id]
            
            # Clear user temp files
            user_dir = Config.TEMP_DIR / str(user.id)
            if user_dir.exists():
                import shutil
                shutil.rmtree(user_dir)
            
            await update.message.reply_text(
                "✅ **تم المسح بنجاح**\n"
                "يمكنك إرسال ملف جديد"
            )
            
        except Exception as e:
            logger.error(f"Clear error: {e}")
            await update.message.reply_text("❌ **حدث خطأ أثناء المسح**")
    
    def calculate_file_hash(self, file_path: str) -> str:
        """حساب بصمة الملف"""
        try:
            hasher = hashlib.sha256()
            
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(4096), b''):
                    hasher.update(chunk)
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Hash error: {e}")
            return str(int(time.time()))
    
    def run(self):
        """تشغيل البوت"""
        self.app = Application.builder().token(self.token).build()
        
        # Add handlers
        self.app.add_handler(CommandHandler("start", self.start))
        self.app.add_handler(CommandHandler("stats", self.stats_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("clear", self.clear_command))
        
        self.app.add_handler(MessageHandler(filters.Document.ALL, self.handle_document))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_text))
        
        # Run bot
        print("\n" + "="*60)
        print("🚀 ULTRA FILE SEARCH BOT v2.0")
        print("="*60)
        print(f"✅ Bot Token: {self.token[:10]}...")
        print(f"👑 Admin ID: {Config.ADMIN_IDS[0]}")
        print(f"📁 Max File Size: {Config.MAX_FILE_SIZE / 1024 / 1024 / 1024:.1f} GB")
        print("="*60)
        print("✅ Bot is running...")
        print("="*60)
        
        self.app.run_polling(allowed_updates=Update.ALL_UPDATES)

# ==================== MAIN ====================
def main():
    """الدالة الرئيسية"""
    # Setup configuration
    Config.setup()
    
    try:
        # Create and run bot
        bot = UltraFileSearchBot(Config.BOT_TOKEN)
        bot.run()
        
    except KeyboardInterrupt:
        print("\n🛑 تم إيقاف البوت")
        
    except Exception as e:
        print(f"❌ خطأ: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()