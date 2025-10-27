import { useState } from "react";
import axios from "axios";
import { API_BASE_URL } from "./constants";

const api = axios.create({
  baseURL: API_BASE_URL,
  withCredentials: true,
});

function App() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [msg, setMsg] = useState("");

  const register = async () => {
    await api.post("/register", { email, password });
    setMsg("Registered");
  };

  const login = async () => {
    await api.post("/login", { email, password });
    setMsg("Logged in");
  };

  const upload = async () => {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    const res = await api.post("/upload", form, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    setMsg("Uploaded dataset " + res.data.dataset_id);
  };

  // test backend connection
  const testConnection = async () => {
    const res = await api.get("/test");
    setMsg(res.data);
  };

  return (
    <div style={{ padding: 20 }}>
      <h2>ML Web App</h2>
      <input placeholder="email" value={email} onChange={(e) => setEmail(e.target.value)} />
      <input placeholder="password" value={password} onChange={(e) => setPassword(e.target.value)} type="password" />
      <button onClick={register}>Register</button>
      <button onClick={login}>Login</button>
      <hr />
      <input type="file" onChange={(e) => setFile(e.target.files?.[0] || null)} />
      <button onClick={upload}>Upload</button>
      <p>{msg}</p>
    </div>
  );
}

export default App;
