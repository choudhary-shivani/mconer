query_dict = {
    "selected_query_literary_work": """SELECT ?work ?workLabel ?workDesc WHERE
{
  ?work wdt:P31/wdt:P279* wd:Q7725634  . # instance of any subclass of work of art
  ?work  rdfs:label ?workLabel .
  ?work schema:description ?workDesc  .
   FILTER ( lang(?workLabel) = "en" ).
    FILTER ( lang(?workDesc) = "en" )
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }

}"""
}
"""SELECT ?work ?workLabel ?workDesc WHERE
{
  ?work wdt:P106/wdt:P279* wd:Q2259532  . # instance of any subclass occupation cleric
  ?work  rdfs:label ?workLabel .
  ?work schema:description ?workDesc  .
   FILTER ( lang(?workLabel) = "en" ).
    FILTER ( lang(?workDesc) = "en" )
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }

}LIMIT 200000"""

"""
SELECT ?work ?workLabel ?workDesc WHERE
{
  ?work wdt:P31/wdt:P279* wd:Q11460  . # instance of any subclass of work of cpthing
  ?work  rdfs:label ?workLabel .
  ?work schema:description ?workDesc  .
   FILTER ( lang(?workLabel) = "en" ).
    FILTER ( lang(?workDesc) = "en" )
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }

}LIMIT 200000
"""
"""
SELECT ?work ?workLabel ?workDesc WHERE
{
  ?work wdt:P31/wdt:P279* wd:Q12819564  . # instance of any subclass of work of station
#   ?work wdt:P373 wd:
  ?work  rdfs:label ?workLabel .
  ?work schema:description ?workDesc  .
   FILTER ( lang(?workLabel) = "en" ).
    FILTER ( lang(?workDesc) = "en" )
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }

}LIMIT 200000
"""
"""
SELECT ?work ?workLabel ?workDesc WHERE
{
  ?work wdt:P31/wdt:P279* wd:Q786820  . # instance of any subclass of work of car
#   ?work wdt:P373 wd:
  ?work  rdfs:label ?workLabel .
  ?work schema:description ?workDesc  .
   FILTER ( lang(?workLabel) = "en" ).
    FILTER ( lang(?workDesc) = "en" )
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }

}LIMIT 200000
"""
"""
SELECT ?work ?workLabel ?workDesc WHERE
{
  ?work wdt:P106/wdt:P279* wd:Q1028181  . # instance of any subclass of work of artist
#   ?work wdt:P373 wd:
  ?work  rdfs:label ?workLabel .
  ?work schema:description ?workDesc  .
   FILTER ( lang(?workLabel) = "en" ).
    FILTER ( lang(?workDesc) = "en" )
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }

}LIMIT 200000
"""
"""
SELECT ?work ?workLabel ?workDesc WHERE
{
  ?work wdt:P31/wdt:P279* wd:Q169872  . # instance of any subclass of work of symptoms
#   ?work wdt:P373 wd:
  ?work  rdfs:label ?workLabel .
  ?work schema:description ?workDesc  .
   FILTER ( lang(?workLabel) = "en" ).
    FILTER ( lang(?workDesc) = "en" )
#   SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }

}LIMIT 200000
"""

"""

"""