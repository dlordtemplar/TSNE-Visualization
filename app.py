from flask import Flask

import common
import neuron
import pair
import passage

# create and configure the app
app = Flask(__name__)
app.config.from_pyfile('config.cfg', silent=False)

app.register_blueprint(common.bp)
app.register_blueprint(neuron.bp)
app.register_blueprint(pair.bp)
app.register_blueprint(passage.bp)

if __name__ == '__main__':
    app.run()
