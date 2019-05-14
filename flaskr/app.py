from flask import Flask

# create and configure the app
app = Flask(__name__, instance_relative_config=True)
app.config.from_mapping(
    SECRET_KEY='dev',
)

from . import common
app.register_blueprint(common.bp)

from . import neuron
app.register_blueprint(neuron.bp)

from . import pair
app.register_blueprint(pair.bp)

from . import passage
app.register_blueprint(passage.bp)

if __name__ == '__main__':
    app.run()
