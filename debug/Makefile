.PHONY: frontend build clean

frontend:
	cd frontend && npm install && npm run build

build: frontend
	cd .. && go build ./debug/...

clean:
	rm -rf frontend/dist frontend/node_modules
