"""Independent finite-set checks of byte-union/prefix semantics and TCP wrap."""
import random
from byte_geometry import insert, Prefix, analyze


def main():
    rng=random.Random(20261119)
    operations=0
    for _ in range(30):
        union=[]
        brute=set()
        for _ in range(20):
            a=rng.randrange(-20,40); b=a+rng.randrange(0,20)
            expected=set(range(a,b))-brute
            novel=insert(union,a,b)
            assert {v for l,r in novel for v in range(l,r)}==expected
            brute.update(range(a,b))
            prefix=Prefix(union)
            assert prefix.total==len(brute)
            for x in (-30,-5,0,15,40,65):
                assert prefix.at(x)==sum(v<x for v in brute)
            operations+=1
    p=[(1,1,0,2**32-6,0,2,0,0,40),(2,2,1,500,2**32-5,18,0,0,40),
       (3,3,0,2**32-5,501,24,10,10,50),(4,4,1,501,5,24,20,20,60)]
    a=analyze(p)
    assert a['geometry_eligible'] and a['unique_payload_bytes']==[10,20]
    assert a['ack_covered_pair_fraction']==1 and a['ack_map_exactly_reconstructed_from_direction_bursts']
    # A genuine hole is not filled by an ACK or normalization.
    hole=p[:3]+[(4,4,0,10,501,24,5,5,45),p[-1]]
    b=analyze(hole)
    assert b['gap_bytes'][0]==5 and not b['geometry_eligible']
    print(f'{operations} finite-set union checks; prefix, wrap and missing-byte checks passed.')


if __name__=='__main__':main()
