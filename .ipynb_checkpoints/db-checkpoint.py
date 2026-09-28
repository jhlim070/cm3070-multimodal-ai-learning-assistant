import sqlite3

# Create User profile and interactions table (learner memory) 
def init_db():
    conn = sqlite3.connect('learner.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS user_profile
                 (user_id TEXT PRIMARY KEY,
                  learning_style TEXT,
                  difficulty TEXT,
                  topics TEXT)''')
    c.execute('''CREATE TABLE IF NOT EXISTS interactions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id TEXT,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  query TEXT,
                  answer TEXT,
                  topic TEXT,
                  feedback INTEGER)''')
    conn.commit()
    conn.close()

# read/write for the profile
def get_profile(user_id="demo"):
    conn = sqlite3.connect('learner.db')
    c = conn.cursor()
    c.execute("SELECT * FROM user_profile WHERE user_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {"learning_style": row[1], "difficulty": row[2], "topics": row[3].split(",")}
    return None

def save_profile(user_id, learning_style, difficulty, topics):
    conn = sqlite3.connect('learner.db')
    c = conn.cursor()
    c.execute("INSERT OR REPLACE INTO user_profile VALUES (?,?,?,?)",
              (user_id, learning_style, difficulty, ",".join(topics)))
    conn.commit()
    conn.close()

# Create new row of interaction
def log_interaction(user_id, query, answer, topic, feedback=None):
    conn = sqlite3.connect('learner.db')
    c = conn.cursor()
    c.execute("INSERT INTO interactions (user_id, query, answer, topic, feedback) VALUES (?,?,?,?,?)",
              (user_id, query, answer, topic, feedback))
    conn.commit()
    conn.close()

# Personalisation engine
def get_weak_topics(user_id, threshold=3):
    """Topics with most queries but low average feedback are 'weak'."""
    conn = sqlite3.connect('learner.db')
    c = conn.cursor()
    c.execute("""SELECT topic, COUNT(*), AVG(feedback) 
                 FROM interactions 
                 WHERE user_id=? AND feedback IS NOT NULL 
                 GROUP BY topic 
                 HAVING COUNT(*) >= ?""", (user_id, threshold))
    rows = c.fetchall()
    conn.close()
    return [row for row in rows if row[2] and row[2] < 3]  # feedback 1-2 means low