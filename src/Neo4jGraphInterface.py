#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 16:18:17 2020

@author: michael
"""

class N4JGraph():
    
    def __init__(self, url, user, pw):
        self.driver = GraphDatabase.driver(url, auth=(user, pw))
    
    def _add_node(self, tx, node_type, attrib_dict):
        
        c = "CREATE (n:" + node_type + ")\n"
        for key in attrib_dict.keys():
            c = c + "SET n." + key + "='" + str(attrib_dict[key]) + "'\n"
        c = c + "RETURN id(n)" 
        
        return tx.run(c).single()[0]
        
        
    def _add_edge(self, tx, id_from, id_to, relationship_type, attrib_dict):
        
        q = ("MATCH (s), (d)" + "\n" +
        "WHERE ID(s) = " + str(id_from) + "\n" +
        "AND ID(d) = " + str(id_to) + "\n" +
        "WITH s, d" + "\n" +
        "CREATE (s)-[r:" + relationship_type + "]->(d)" + "\n")
        
        for key in attrib_dict.keys():
            q = q + "SET r." + key + "=\"" + str(attrib_dict[key]) + "\"\n"
        
        q = q + "RETURN r;"
        
        return tx.run(q).single()[0]
            
    def _remove_node(self, tx, n):
        q = ("MATCH (n)" + "\n" +
             "WHERE ID(n) = " + n + "\n" +
             "DETACH DELETE n;"
            )
        
        return tx.run(q)
    
    def _get_id(self, tx, description):
        
        q = "MATCH (n)\n"
        f = True
        for key in description.keys():
            if f:
                q = q + "WHERE "
            else:
                q = q + "AND "
            q = q + "n."+key + "='" + description[key] + "'\n"
        q = q + "RETURN ID(n);"
            
        return tx.run(q).single()[0]
        
        
        
    
    def add_node(self, node_type, attrib_dict):
        with self.driver.session() as session:
            n = session.write_transaction(self._add_node, node_type, attrib_dict)
            return n
            
    def add_edge(self, source, dest, edge_type, attrib_dict):
        with self.driver.session() as session:
            session.write_transaction(self._add_edge, source, dest, edge_type, attrib_dict)
            return 
    
    def remove_node(self, n):
        with self.driver.session() as session:
            session.write_transaction(self._remove_node, n)
            
    def get_id_from_name(self, name):
        description = {"name":name}
        with self.driver.session() as session:
            i = session.write_transaction(self._get_id, description)
            return i
            
g = N4JGraph("bolt://localhost:7687/", "neo4j", "password")

#n1 = g.add_node("TestNode", {'thing':123456, "thing2":3.14159})
#n2 = g.add_node("TestNode", {'thing':654321, "thing2":3.14159})
#print(n1)
#print(n2)
#g.add_edge(n1, n2, "FOOS", {'a':'bar', 'bar':'baz'})
#g.remove_node("87")
