import mysql.connector
from mysql.connector import Error
from config.config import load_config
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Database:
    def __init__(self):
        self.config = load_config()
        self.connection = None
        self.ensure_database_exists()
        self.connect()

    def ensure_database_exists(self):
        try:
            temp_connection = mysql.connector.connect(
                host=self.config['database']['host'],
                user=self.config['database']['user'],
                password=self.config['database']['password'],
                port=int(self.config['database']['port'])
            )
            cursor = temp_connection.cursor()
            cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.config['database']['database']}")
            cursor.close()
            temp_connection.close()
            logger.info("Database checked/created successfully")
        except Error as e:
            logger.error(f"Error ensuring database exists: {e}")
            raise

    def connect(self):
        try:
            self.connection = mysql.connector.connect(
                host=self.config['database']['host'],
                user=self.config['database']['user'],
                password=self.config['database']['password'],
                database=self.config['database']['database'],
                port=int(self.config['database']['port'])
            )
            logger.info("Connected to MySQL database")
        except Error as e:
            logger.error(f"Error connecting to MySQL: {e}")
            raise

    def execute_query(self, query, params=None, fetch=False):
        cursor = None
        try:
            if not self.connection.is_connected():
                self.connect()
                
            cursor = self.connection.cursor(dictionary=True)
            cursor.execute(query, params or ())
            
            if fetch:
                result = cursor.fetchall()
                return result
            else:
                self.connection.commit()
                return cursor.rowcount
                
        except Error as e:
            logger.error(f"Database error: {e}")
            if self.connection:
                self.connection.rollback()
            raise
        finally:
            if cursor:
                cursor.close()

    def initialize_database(self):
        """Create tables if they don't exist"""
        try:
            # Create customerdetails table
            self.execute_query("""
                CREATE TABLE IF NOT EXISTS customerdetails (
                    CustomerID INT AUTO_INCREMENT PRIMARY KEY,
                    Name VARCHAR(100) NOT NULL,
                    Email_ID VARCHAR(100) NOT NULL UNIQUE,
                    Password VARCHAR(255) NOT NULL,
                    LastLoginTime DATETIME,
                    CreatedAt DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create chathistory table
            self.execute_query("""
                CREATE TABLE IF NOT EXISTS chathistory (
                    ChatID INT AUTO_INCREMENT PRIMARY KEY,
                    CustomerID INT NOT NULL,
                    Question TEXT NOT NULL,
                    Answer TEXT NOT NULL,
                    Timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (CustomerID) REFERENCES customerdetails(CustomerID)
                )
            """)
            
            logger.info("Database tables initialized successfully")
            return True
        except Error as e:
            logger.error(f"Error initializing database: {e}")
            return False

    def create_user(self, name, email, password):
        """Register a new user"""
        try:
            result = self.execute_query(
                "INSERT INTO customerdetails (Name, Email_ID, Password) VALUES (%s, %s, %s)",
                (name, email, password)
            )
            return result > 0
        except Error as e:
            logger.error(f"Error creating user: {e}")
            return False

    def get_user_by_email(self, email):
        """Get user by email"""
        try:
            users = self.execute_query(
                "SELECT * FROM customerdetails WHERE Email_ID = %s",
                (email,),
                fetch=True
            )
            return users[0] if users else None
        except Error as e:
            logger.error(f"Error getting user by email: {e}")
            return None

    def update_last_login(self, user_id):
        """Update user's last login time"""
        try:
            self.execute_query(
                "UPDATE customerdetails SET LastLoginTime = %s WHERE CustomerID = %s",
                (datetime.now(), user_id)
            )
            return True
        except Error as e:
            logger.error(f"Error updating last login: {e}")
            return False

    def save_chat_history(self, user_id, question, answer):
        """Save chat message to history"""
        try:
            self.execute_query(
                "INSERT INTO chathistory (CustomerID, Question, Answer) VALUES (%s, %s, %s)",
                (user_id, question, answer)
            )
            return True
        except Error as e:
            logger.error(f"Error saving chat history: {e}")
            return False

    def get_chat_history(self, user_id, limit=20):
        """Get user's chat history"""
        try:
            return self.execute_query(
                "SELECT Question, Answer, Timestamp FROM chathistory WHERE CustomerID = %s ORDER BY Timestamp DESC LIMIT %s",
                (user_id, limit),
                fetch=True
            )
        except Error as e:
            logger.error(f"Error getting chat history: {e}")
            return []

    def close(self):
        if self.connection and self.connection.is_connected():
            self.connection.close()
            logger.info("Database connection closed")

# Initialize database connection when module is imported
db = Database()
db.initialize_database()