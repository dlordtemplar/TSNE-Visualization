from flask import Flask

import common
import neuron
import pair
import live
import passage
import questions
import pos
import tsne_neurons

# create and configure the app
app = Flask(__name__)
app.config.from_pyfile('config.cfg', silent=False)

app.register_blueprint(common.bp)
app.register_blueprint(neuron.bp)
app.register_blueprint(pair.bp)
app.register_blueprint(live.bp)
app.register_blueprint(passage.bp)
app.register_blueprint(questions.bp)
app.register_blueprint(pos.bp)
app.register_blueprint(tsne_neurons.bp)

if __name__ == '__main__':
    app.run()
