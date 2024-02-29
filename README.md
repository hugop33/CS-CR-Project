# Connaissances et Raisonnement - Recommandation de films
*Aymeric PALARIC, Hugo PLOTTU, Alexandre PETIT.*

## Problématique
Comment recommander des films à un utilisateur en fonction de ses précédentes recherches et intéractions ?


## Méthode
**Base de données** : la BDD utilisée est celle issue du TD4 (IMDb). Elle contient des informations sur des films, des acteurs, des réalisateurs, des genres, des notes.
Cette base ne représente qu'une cinquantaine de films, mais elle est suffisante pour illustrer le principe de recommandation.
L'utilisation de la BDD IMDb complète ne nous a pas paru utile, et aurait été trop lourde pour les calculs.
Nous avons ensuite enrichi cette BDD avec des utilisateurs, puis peuplé avec des requêtes (recherches, likes, dislikes) aléatoires pour simuler des utilisateurs.

**Recommandation** : pour recommander des films à un utilisateur, nous attribuons un score à chaque film en fonction de ses intéractions avec les films précédents. Le calcul de ce score repose sur la logique suivante :
- Si l'utilisateur a aimé un film, il est probable qu'il aime les films du même genre, avec les mêmes acteurs, réalisateurs, etc. On donne donc un score +3 aux films qui ont des points communs avec les films aimés.
- Si l'utilisateur a détesté un film, on donne un score -3 aux films qui ont des points communs avec les films détestés.
- Si l'utilisateur a simplement regardé un film, on donne un score +1 aux films qui ont des points communs avec les films regardés.

## Utilisation
Pour utiliser ce programme, si la base de données entière décrite précédemment existe déjà (avec les requêtes utilisateurs, normalement déjà incluse dans le zip), il suffit de lancer le fichier `main.py` puis de donner un identifiant d'utilisateur (parmi ceux existants dans la BDD) pour obtenir une recommandation de films.

Si la base de données n'existe pas, il faut modifier la constante `GENERATE_DB` dans le fichier `config.py` pour la passer à `True`, puis lancer le fichier `main.py`. La base de données sera alors générée, et les requêtes utilisateurs seront peuplées aléatoirement. Ensuite, on pourra donner un identifiant d'utilisateur pour obtenir une recommandation de films.