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
