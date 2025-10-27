import { useState } from "react";
import axios from "axios";

function App() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [msg, setMsg] = useState("");

  const register = async () => {
    await axios.post("/api/register", { email, password }, { withCredentials: true });
    setMsg("Registered");
  };

  const login = async () => {
    await axios.post("/api/login", { email, password }, { withCredentials: true });
    setMsg("Logged in");
  };

  const upload = async () => {
    if (!file) return;
    const form = new FormData();
    form.append("file", file);
    const res = await axios.post("/api/upload", form, {
      withCredentials: true,
      headers: { "Content-Type": "multipart/form-data" },
    });
    setMsg("Uploaded dataset " + res.data.dataset_id);
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
