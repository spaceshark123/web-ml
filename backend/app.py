from endpoints import app, socketio

# ===== Main =====
if __name__ == '__main__':
    socketio.run(app, port=5000, debug=True, allow_unsafe_werkzeug=True)
