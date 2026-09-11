package api

import (
	"encoding/json"
	"errors"
	"net/http"
)

// errorEnvelope matches the OpenAI client error shape so OpenAI-compatible
// SDKs render orla errors correctly without special handling.
type errorEnvelope struct {
	Error errorBody `json:"error"`
}

type errorBody struct {
	Message string `json:"message"`
	Type    string `json:"type"`
}

func errorType(status int) string {
	switch status {
	case http.StatusBadRequest, http.StatusUnprocessableEntity, http.StatusUnsupportedMediaType:
		return "invalid_request_error"
	case http.StatusUnauthorized, http.StatusForbidden:
		return "permission_denied"
	case http.StatusNotFound:
		return "not_found"
	case http.StatusTooManyRequests:
		return "rate_limit_exceeded"
	case http.StatusRequestEntityTooLarge:
		return "invalid_request_error"
	case http.StatusServiceUnavailable:
		return "server_error"
	default:
		if status >= 500 {
			return "server_error"
		}
		return "api_error"
	}
}

// writeError renders an error envelope. The message is the unwrapped err
// string, the status drives the type field.
func writeError(w http.ResponseWriter, status int, err error) {
	msg := http.StatusText(status)
	if err != nil {
		msg = err.Error()
	}
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(errorEnvelope{
		Error: errorBody{Message: msg, Type: errorType(status)},
	})
}

// writeErrorMsg is the string variant of writeError, for places where
// constructing an error{} just for its message would be needlessly
// verbose.
func writeErrorMsg(w http.ResponseWriter, status int, msg string) {
	writeError(w, status, errors.New(msg))
}

// decodeJSON parses the request body into v. The body is left limited
// by the bodyLimit middleware upstream, we just translate decode errors
// into a 400.
func decodeJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		writeError(w, http.StatusBadRequest, err)
		return false
	}
	return true
}
