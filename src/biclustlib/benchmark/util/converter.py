import asyncio
from os.path import join, dirname

import aiohttp


class Converter:
    def __init__(self, gene_mapping_path: str = None):
        """
        This is a utility class used to asynchronously convert SGD Systematic Names to SGD IDs,
        which is required to perform GO Enrichment Analysis.

        :param gene_mapping_path: A path to an optional mapping file,
         containing a dictionary of SGD Systematic Names and their respective IDs.
        """

        self.gene_mapping = {}
        self.gene_mapping_path = gene_mapping_path
        if self.gene_mapping_path:
            try:
                f = open(self.gene_mapping_path)
                for line in f:
                    sysname, sgdid = line.split()
                    self.gene_mapping[sysname] = sgdid
                f.close()
            except OSError:
                raise ValueError(f'File {self.gene_mapping_path} does not exist')

    async def convert(self, sysname_list: list) -> list:
        """
        Convert a list of SGD Systematic Names to SGD IDs.
        Works asynchronously, must be awaited.
        Unknown genes are translated at the rate of around 10 genes per second.

        :param sysname_list: A list of gene SGD Systematic Names.
        :return: A list of respective SGD IDs.
        """
        async with aiohttp.ClientSession('https://www.yeastgenome.org') as session:
            ret = []
            batch_size = 500
            for i in range(0, len(sysname_list), batch_size):
                ret.extend(
                    await asyncio.gather(
                        *[self._get_sgd_id(sysname, session) for sysname in sysname_list[i:i + batch_size]]))
            return ret

    async def _get_sgd_id(self, sysname: str, session: aiohttp.ClientSession) -> str:
        sysname = sysname.strip()
        if sysname in self.gene_mapping:
            return self.gene_mapping[sysname]
        else:
            async with session.get(f'/backend/locus/{sysname}') as response:
                try:
                    ret = await response.json()
                    self.gene_mapping[sysname] = ret['sgdid']
                    return ret['sgdid']
                except aiohttp.ContentTypeError:
                    return 'OBSOLETE'

    def dump_gene_mapping(self):
        """
        Update the gene mapping with new entries.
        """
        f = open(self.gene_mapping_path, mode='w')
        for sysname in sorted(self.gene_mapping):
            f.write(sysname + ' ' + self.gene_mapping[sysname] + '\n')
        f.close()
