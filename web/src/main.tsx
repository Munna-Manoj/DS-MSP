import React from "react";
import ReactDOM from "react-dom/client";
import { App } from "./App";
import "./index.css";
import { selfCheck } from "./lib/cameras";

if (import.meta.env.DEV) {
  const err = selfCheck();
  console.log(
    `[ds-msp] all-model project∘unproject self-check — worst error: ${err.toExponential(2)} rad (expect ~1e-7)`,
  );
}

ReactDOM.createRoot(document.getElementById("root")!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>,
);
