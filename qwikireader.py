import json
import pandas as pd
from qwikidata.entity import WikidataItem, WikidataProperty
from qwikidata.sparql import return_sparql_query_results
from qwikidata.linked_data_interface import get_entity_dict_from_api
from qwikidata.entity import WikidataItem, WikidataProperty, WikidataLexeme
from qwikiquery import *
type_to_entity_class = {"item": WikidataItem, "property": WikidataProperty}
max_entities = 5
entities = []

query_string = """
SELECT ?minister ?ministerLabel WHERE {
  ?minister wdt:P31 wd:Q5 .
  SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
} LIMIT 50
"""
query_string_art = """
SELECT ?work ?workLabel ?workDesc WHERE
{
  ?work wdt:P31/wdt:P279* wd:Q11460  . # instance of any subclass of work of art
  ?work  rdfs:label ?workLabel .
  ?work schema:description ?workDesc  .
   FILTER ( lang(?workLabel) = "en" ).
    FILTER ( lang(?workDesc) = "en" )
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }

}LIMIT 200000
"""
# query_string_art = """SELECT ?work ?workLabel
# WHERE
# {
#   ?work wdt:P31/wdt:P279* wd:Q838948. # instance of any subclass of work of art
# #   SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE]". }
#   ?work rdfs:label ?workLabel. FILTER( LANG(?workLabel)="en" )
# }LIMIT 2"""

results = return_sparql_query_results(query_string)
with open('extract.json', 'w') as f:
    json.dump(results, f)
# print(type(results))
x = pd.DataFrame.from_dict(results['results']['bindings'])
x['wlabelCleaned'] = x['workLabel'].apply(lambda x: x['value'])
x['wdescCleaned'] = x['workDesc'].apply(lambda x: x['value'])
x.to_csv('extracted-clothing.csv', index=False)
# print(WikidataItem(get_entity_dict_from_api('Q3660440')))