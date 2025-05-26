# Face Recognition Authentication System

A secure face recognition authentication system built with Flask and face_recognition library.

## Features

- Face-based user registration and login
- Secure password hashing
- Account lockout after failed attempts
- Email notifications for suspicious login attempts
- Real-time webcam face detection
- Modern UI with Tailwind CSS

## Prerequisites

- Python 3.10 or higher
- Visual C++ Build Tools (for Windows)
- Webcam

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd face_recognition_app
```

2. Install Visual C++ Build Tools (Windows only):
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "Desktop development with C++"

3. Create and activate virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Create .env file:
```bash
SECRET_KEY=your_secret_key
MAIL_SERVER=smtp.gmail.com
MAIL_PORT=587
MAIL_USERNAME=your_email@gmail.com
MAIL_PASSWORD=your_app_password
```

6. Run the application:
```bash
python app.py
```

## Usage

1. Register a new account with your email, password, and face
2. Login using your email, password, and face verification
3. Face verification requires good lighting and clear face visibility

## Security Features

- Face distance threshold for accurate matching
- Multiple face samples for better recognition
- Account lockout after 3 failed attempts
- Email alerts for suspicious login attempts
- Secure password hashing with bcrypt
- Session management with JWT

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/) 