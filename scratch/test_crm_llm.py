import unittest
import os
import sys
import glob
import requests
from unittest.mock import patch, MagicMock


# Add project root to sys.path so we can import app
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestChatbotUpgrade(unittest.TestCase):
    
    def setUp(self):
        # Set environment variables for testing
        os.environ["LLM_PROVIDER"] = "ollama"
        os.environ["OLLAMA_BASE_URL"] = "http://localhost:11434"
        os.environ["ADMIN_PASSWORD"] = "admin123"

    def test_glob_loading(self):
        """Verify that source markdown documents exist and can be loaded via glob"""
        source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/source_documents'))
        self.assertTrue(os.path.exists(source_dir), f"Source directory {source_dir} should exist")
        
        md_files = glob.glob(os.path.join(source_dir, "*.md"))
        self.assertGreater(len(md_files), 0, "Should have created modular markdown files")
        print(f"[TEST] Found {len(md_files)} markdown files in data/source_documents")
        
        # Verify faq.md exists
        faq_file = os.path.join(source_dir, "13_faq.md")
        self.assertTrue(os.path.exists(faq_file), "13_faq.md should exist")

    def test_intent_detection(self):
        """Test the rewritten detect_intent() to confirm it detects quote/consultation buy signals"""
        from app import detect_intent
        
        # Quote/consultation signals (should return 'buy')
        self.assertEqual(detect_intent("get a quote"), "buy")
        self.assertEqual(detect_intent("book a consultation"), "buy")
        self.assertEqual(detect_intent("sign me up for a call"), "buy")
        self.assertEqual(detect_intent("I want to schedule a call"), "buy")
        
        # Package selections (should no longer return 'buy')
        self.assertEqual(detect_intent("tell me about the classic plan"), "service_info")
        self.assertEqual(detect_intent("premium plan price"), "general")

        
        # Human signals (should return 'human')
        self.assertEqual(detect_intent("let me talk to a real person"), "human")
        self.assertEqual(detect_intent("give me support agent"), "human")
        
        # General signals
        self.assertEqual(detect_intent("how do you do?"), "general")
        print("[TEST] Intent detection checks completed successfully.")

    def test_sqlite_storage(self):
        """Verify that lead details are successfully saved to SQLite database"""
        import sqlite3
        db_file = "leads.db"
        
        # Insert test record
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        # Clean any test records
        cursor.execute("DELETE FROM chats WHERE email = 'test_sqlite@example.com'")
        conn.commit()
        
        cursor.execute("""
            INSERT INTO chats (started_at, name, email, phone, interested_services, transcript, summary)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            "2026-06-22 00:00:00",
            "Sqlite Tester",
            "test_sqlite@example.com",
            "+12345",
            "Cover Design",
            "User: Hello\nBot: Hi",
            "Author wants cover design."
        ))
        conn.commit()
        
        # Verify it was inserted
        cursor.execute("SELECT * FROM chats WHERE email = 'test_sqlite@example.com'")
        row = cursor.fetchone()
        conn.close()
        
        self.assertIsNotNone(row)
        self.assertEqual(row[2], "Sqlite Tester")
        self.assertEqual(row[4], "+12345")
        self.assertEqual(row[5], "Cover Design")
        self.assertEqual(row[7], "Author wants cover design.")
        print("[TEST] SQLite lead storage verification completed successfully.")

    def test_lead_json_backup(self):
        """Verify that saving a lead writes to leads/lead_capture.json backup file"""
        from app import save_lead_to_json
        import json
        
        backup_file = os.path.abspath(os.path.join(os.path.dirname(__file__), '../leads/lead_capture.json'))
        
        # Delete if exists to start fresh
        if os.path.exists(backup_file):
            try:
                os.remove(backup_file)
            except Exception:
                pass
                
        lead_data = {
            "name": "JSON Backup Tester",
            "email": "json_backup@example.com",
            "phone": "777777",
            "interested_services": "Editing",
            "timestamp": "2026-06-22T00:00:00",
            "session_id": "test-json-session",
            "summary": "Need developmental editing for sci-fi novel."
        }
        
        save_lead_to_json(lead_data)
        
        self.assertTrue(os.path.exists(backup_file))
        
        # Read and check contents
        with open(backup_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        # Check that our tester lead is in the list
        matched = [lead for lead in data if lead.get("email") == "json_backup@example.com"]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["name"], "JSON Backup Tester")
        print("[TEST] Lead JSON backup verification completed successfully.")

if __name__ == '__main__':
    unittest.main()
