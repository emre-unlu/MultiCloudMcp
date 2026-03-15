package main

import (
	"bytes"
	"encoding/json"
	"io"
	"log"
	"net/http"
	"os"
	"time"
)

type runReq struct {
	Goal string `json:"goal"`
}
type runResp map[string]any // pass-through JSON

func main() {
	supervisorURL := getenv("SUPERVISOR_URL", "http://127.0.0.1:9000/run")
	addr := getenv("UI_ADDR", ":8088")
	client := &http.Client{Timeout: supervisorTimeout()}

	mux := http.NewServeMux()

	// Health
	mux.HandleFunc("/api/health", func(w http.ResponseWriter, r *http.Request) {
		writeJSON(w, http.StatusOK, map[string]any{
			"ok":  true,
			"sup": supervisorURL,
		})
	})

	// Proxy /api/run -> SUPERVISOR_URL
	mux.HandleFunc("/api/run", func(w http.ResponseWriter, r *http.Request) {
		enableCORS(w, r)
		if r.Method == http.MethodOptions {
			w.WriteHeader(http.StatusNoContent)
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		body, err := io.ReadAll(r.Body)
		if err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}

		// forward to supervisor
		req, err := http.NewRequest(http.MethodPost, supervisorURL, bytes.NewReader(body))
		if err != nil {
			http.Error(w, "upstream error (build)", http.StatusBadGateway)
			return
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := client.Do(req)
		if err != nil {
			http.Error(w, "upstream error (connect): "+err.Error(), http.StatusBadGateway)
			return
		}
		defer resp.Body.Close()

		out, err := io.ReadAll(resp.Body)
		if err != nil {
			http.Error(w, "upstream error (read)", http.StatusBadGateway)
			return
		}
		// Pass-through status + body
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(resp.StatusCode)
		w.Write(out)
	})

	// Static UI
	fs := http.FileServer(http.Dir("./web"))
	mux.Handle("/", fs)

	log.Printf("UI: http://127.0.0.1%s  (proxying to SUPERVISOR_URL=%s)", addr, supervisorURL)
	log.Fatal(http.ListenAndServe(addr, withCORS(mux)))
}

func enableCORS(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Access-Control-Allow-Origin", "*")
	w.Header().Set("Access-Control-Allow-Headers", "Content-Type, Authorization")
	w.Header().Set("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
}
func withCORS(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		enableCORS(w, r)
		next.ServeHTTP(w, r)
	})
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	_ = json.NewEncoder(w).Encode(v)
}

func supervisorTimeout() time.Duration {
	const fallback = 5 * time.Minute
	val := getenv("SUPERVISOR_TIMEOUT", "5m")
	d, err := time.ParseDuration(val)
	if err != nil {
		log.Printf("Invalid SUPERVISOR_TIMEOUT %q, defaulting to %s", val, fallback)
		return fallback
	}
	if d <= 0 {
		return 0
	}
	return d
}
func getenv(k, def string) string {
	if v := os.Getenv(k); v != "" {
		return v
	}
	return def
}
