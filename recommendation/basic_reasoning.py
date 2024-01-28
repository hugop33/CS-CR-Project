from utils.film_ontology import load_ontology
from utils.user_ontology import add_request, save_ontology, random_populating_from_filmontology
from config import DATA
from owlready2 import sync_reasoner_pellet


def instantiate_ontologies():
    fonto = load_ontology(f'{DATA}smallIMDB.owl')
    uonto = create_ontology()
