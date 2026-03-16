"""
StructureDCA
============

**Structure-Informed Direct Coupling Analysis** to predict
the **effects of missense mutations on proteins**.
It incorporates the residues contact map derived from protein
3D structures to infer an **evolutionary sparse DCA model**.
This approach leverages the observation that functionally relevant,
co-evolving residues are most often in structural contact.

Example of usage in Python:

>>> from structuredca import StructureDCA
>>> sdca = StructureDCA('./msa1.fasta', './pdb1.pdb', 'A')
>>> mutation_score = sdca.eval_mutation('K24M:H39G', reweight_by_rsa=False)
>>> all_muts_scores_table = sdca.eval_mutations_table()
"""

from structuredca.structuredca import StructureDCA
