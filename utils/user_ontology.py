import owlready2
from owlready2 import Thing, get_ontology, AllDisjoint, sync_reasoner_pellet, Imp, ObjectProperty
import json
import csv
import random

from config import OWLREADY2_JAVA_EXE, DATA

# Set the path to the Java executable
owlready2.JAVA_EXE = OWLREADY2_JAVA_EXE

DUMB_USER_NAMES = ["user1", "user2", "user3", "user4", "user5", "user6", "user7", "user8", "user9", "user10"]
USER_IDS = [f"u{i}" for i in range(1, 11)]

def create_ontology():
    onto = get_ontology("http://students.org/ontologies/user.owl")
    return onto

def load_ontology(filename):
    onto = get_ontology(filename).load()
    return onto


def save_ontology(onto, filename):
    onto.save(file=filename, format="rdfxml")


def create_classes(onto):
    with onto:
        class User(Thing): pass
        class Film(Thing): pass
        class Personne(Thing): pass
        class GenreFilm(Thing): pass
        class Pays(Thing): pass

        AllDisjoint([Personne, Film, GenreFilm, Pays, User])

        class Acteur(Personne): pass
        class Réalisateur(Personne): pass
        class Scénariste(Personne):  pass

        class Acteur(Personne): pass
        class Réalisateur(Personne): pass
        class Scénariste(Personne):  pass

        # créer la relation aCherché qui lie un utilisateur à un attribut de film (acteur, réalisateur, genre, pays, ...)
        class aCherché(ObjectProperty):
            domain = [User]
            range = [Film, Acteur, Réalisateur, GenreFilm, Pays, Scénariste]
        
        # relations inverses
        class CherchéPar(ObjectProperty):
            domain = [Film, Acteur, Réalisateur, GenreFilm, Pays, Scénariste]
            range = [User]
            inverse_property = aCherché


def add_request(onto, user_id, attribute):
    """
    Add a request to the ontology.
    """
    with onto:
        user = onto.User(user_id)
        attribute = onto[attribute]
        user.aCherché.append(attribute)


def add_requests(onto, user_id, attributes):
    """
    Add a list of requests to the ontology.
    """
    with onto:
        user = onto.User(user_id)
        for attribute in attributes:
            attribute = onto[attribute]
            user.aCherché.append(attribute)


def sync_with_filmontology(onto, fonto):
    """
    Synchronize the user ontology with the film ontology.
    """
    with onto:
        films = list(fonto.Film.instances())
        films = [(film.name, onto.Film) for film in films]
        acteurs = list(fonto.Acteur.instances())
        acteurs = [(acteur.name, onto.Acteur) for acteur in acteurs]
        réalisateurs = list(fonto.Réalisateur.instances())
        réalisateurs = [(réalisateur.name, onto.Réalisateur) for réalisateur in réalisateurs]
        genres = list(fonto.GenreFilm.instances())
        genres = [(genre.name, onto.GenreFilm) for genre in genres]
        pays = list(fonto.Pays.instances())
        pays = [(pays.name, onto.Pays) for pays in pays]
        scénaristes = list(fonto.Scénariste.instances())
        scénaristes = [(scénariste.name, onto.Scénariste) for scénariste in scénaristes]
        attributes = films + acteurs + réalisateurs + genres + pays + scénaristes
        for attribute in attributes:
            attribute_instance = onto[attribute[0]] if attribute[0] in onto else attribute[1](attribute[0])
            attribute_instance.is_a.append(attribute[1])


def random_populating(onto, n_req):
    """
    Add n_req random requests to the user ontology.
    Attributes requested by the users are chosen randomly.
    """
    with onto:
        users = [random.choice(USER_IDS) for _ in range(n_req)]
        attributes = [random.choice(list(onto.classes())) for _ in range(n_req)]
        for user_id, attribute in zip(users, attributes):
            user = onto.User(user_id)
            attribute = onto[attribute.name]
            user.aCherché.append(attribute)



if __name__ == "__main__":
    onto = create_ontology()
    create_classes(onto)
    fonto = load_ontology(f'{DATA}smallIMDB.owl')
    sync_with_filmontology(onto, fonto)
    random_populating(onto, 100)
    sync_reasoner_pellet(infer_property_values=True, infer_data_property_values=True)
    save_ontology(onto, f'{DATA}smallIMDB_users.owl')