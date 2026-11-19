# The tensor-core model against the published A100/H100/H200/B200 measurements and SF Tensor's B200
# tcgen05 model, and the MUFU model against PTX Math, each fetched at a pinned commit.
lake build num-tests && sh tools/num/tensor-data.sh && sh tools/num/sftensor-data.sh && sh tools/num/mufu-check.sh
