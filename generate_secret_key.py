#!/usr/bin/env python3
"""
Generate Django secret key and create .env file
"""
import secrets
import string
from pathlib import Path

def generate_secret_key(length=50):
    """Generate a secure Django secret key."""
    alphabet = string.ascii_letters + string.digits + '!@#$%^&*(-_=+)'
    secret_key = ''.join(secrets.choice(alphabet) for _ in range(length))
    return secret_key

def create_env_file():
    """Create .env file with generated secret key."""
    secret_key = generate_secret_key()
    
    env_content = f"""# Django Configuration
DJANGO_SECRET_KEY={secret_key}
DEBUG=True
ALLOWED_HOSTS=localhost,127.0.0.1,your-domain.com

# Model Configuration
MODEL_PATH=./models/

# Deployment Configuration
PORT=8000
WORKERS=1
TIMEOUT=120
"""
    
    env_path = Path('.env')
    
    if env_path.exists():
        response = input(".env file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Cancelled")
            return
    
    with open(env_path, 'w') as f:
        f.write(env_content)
    
    print(f"✅ Created .env file with new secret key")
    print(f"🔑 Secret key: {secret_key}")
    print("\n📝 Next steps:")
    print("1. Load environment variables: pip install python-dotenv")
    print("2. In your settings.py, add:")
    print("   from dotenv import load_dotenv")
    print("   load_dotenv()")
    print("3. For production, set environment variables in your hosting platform")

if __name__ == "__main__":
    print("🔐 Django Secret Key Generator")
    print("=" * 40)
    
    choice = input("Generate new secret key and .env file? (y/N): ")
    if choice.lower() == 'y':
        create_env_file()
    else:
        # Just generate and print a key
        key = generate_secret_key()
        print(f"🔑 Generated secret key: {key}")
        print("\nAdd this to your environment variables:")
        print(f"DJANGO_SECRET_KEY={key}")