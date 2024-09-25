from flask import Flask

def create_app():
    app = Flask(__name__)
    with app.app_context():
        from src.routes.ml_routes import ml_predict
        from src.routes.graph_routes import generate_graph  # Assuming you have this route
        app.register_blueprint(ml_predict)
        app.register_blueprint(generate_graph)
    return app
