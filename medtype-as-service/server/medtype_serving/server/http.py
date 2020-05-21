from multiprocessing import Process, Event
from termcolor import colored
from .helper import set_logger


class MedTypeHTTPProxy(Process):
	def __init__(self, args):
		super().__init__()
		self.args = args
		self.is_ready = Event()

	def create_flask_app(self):
		from flask import Flask, request
		from flask_compress import Compress
		from flask_cors import CORS
		from flask_json import FlaskJSON, as_json, JsonError
		from medtype_serving.client import ConcurrentMedTypeClient

		logger = set_logger(colored('PROXY', 'red'))

		self.bc  = None
		app = Flask(__name__)
		app.config['SWAGGER'] = {
		  'title': 'Colors API',
		  'uiversion': 3,
		  'openapi': '3.0.2'
		}

		@app.route('/status/server', methods=['GET'])
		@as_json
		def get_server_status():
			if self.bc is None:
				self.bc = ConcurrentMedTypeClient(max_concurrency=self.args.http_max_connect, port=self.args.port, port_out=self.args.port_out)
			return self.bc.server_status

		@app.route('/status/client', methods=['GET'])
		@as_json
		def get_client_status():
			if self.bc is None:
				self.bc = ConcurrentMedTypeClient(max_concurrency=self.args.http_max_connect, port=self.args.port, port_out=self.args.port_out)
			return self.bc.status

		@app.route('/run_linker', methods=['POST'])
		@as_json
		def encode_query():
		# support up to 10 concurrent HTTP requests
			if self.bc is None:
				self.bc = ConcurrentMedTypeClient(max_concurrency=self.args.http_max_connect, port=self.args.port, port_out=self.args.port_out)

			data = request.form if request.form else request.json
			try:
				logger.info('new request from %s' % request.remote_addr)
				return {'id': data['id'], 'result': self.bc.run_linker(data['data'])}

			except Exception as e:
				logger.error('error when handling HTTP request', exc_info=True)
				raise JsonError(description=str(e), type=str(type(e).__name__))

		CORS(app, origins=self.args.cors)
		FlaskJSON(app)
		Compress().init_app(app)
		return app

	def run(self):
		app = self.create_flask_app()
		self.is_ready.set()
		app.run(port=self.args.http_port, threaded=True, host='0.0.0.0')
