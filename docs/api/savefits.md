# `drpangloss.savefits`

Legacy module, kept for existing scripts. It re-exports the ImPlaneIA-derived
writer from [`drpangloss.oifits_implaneia`](oifits_implaneia.md) under its old
names. New code should use [`drpangloss.oifits`](oifits.md). It is not
imported by `import drpangloss`; use `import drpangloss.savefits`.

## Functions

::: drpangloss.savefits
    options:
      members:
        - rad2mas
        - GetWavelength
        - Format_STAINDEX_V2
        - Format_STAINDEX_T3
        - ApplyFlag
        - save
        - cp_indices
