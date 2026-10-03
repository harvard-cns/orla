package api

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
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

// writeBodyError renders a failure to read or decode the request body.
// A body over the cap that bodyLimitMiddleware sets returns 413 and
// names the limit, so the client knows how far to shrink it. Any other
// failure is malformed input and returns 400.
func writeBodyError(w http.ResponseWriter, err error) {
	if mbe, ok := errors.AsType[*http.MaxBytesError](err); ok {
		writeErrorMsg(w, http.StatusRequestEntityTooLarge,
			fmt.Sprintf("request body exceeds the %d byte limit", mbe.Limit))
		return
	}
	writeError(w, http.StatusBadRequest, err)
}

// decodeJSON parses the request body into v and reports whether it
// succeeded. On failure it has already written the error response. The
// body must hold exactly one JSON value. Reading it to the end means a
// body over the size cap returns 413 even when its first value fits.
func decodeJSON(w http.ResponseWriter, r *http.Request, v any) bool {
	dec := json.NewDecoder(r.Body)
	dec.DisallowUnknownFields()
	if err := dec.Decode(v); err != nil {
		writeBodyError(w, err)
		return false
	}
	var extra json.RawMessage
	if err := dec.Decode(&extra); !errors.Is(err, io.EOF) {
		if err == nil {
			err = errors.New("request body must hold a single JSON value")
		}
		writeBodyError(w, err)
		return false
	}
	return true
}
