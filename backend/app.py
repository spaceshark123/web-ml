from endpoints import app, socketio
from os import getenv

# ===== Main =====
if __name__ == '__main__':
    port = int(getenv("PORT", 5050))
    socketio.run(app, port=port, debug=True, allow_unsafe_werkzeug=True)
