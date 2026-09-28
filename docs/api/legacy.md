# `drpangloss.legacy`

Legacy OIFITS tools derived from [ImPlaneIA](https://github.com/anand0xff/ImPlaneIA),
working on its dictionary layout (phases in **degrees**). They are not imported
by `import drpangloss`. New code should use [`drpangloss.oifits`](oifits.md) and
[`OIData`](oidata.md).

::: drpangloss.legacy.oifits_implaneia
    options:
      members:
        - save
        - load
        - load_oifits
        - GetWavelength
        - Format_STAINDEX_V2
        - Format_STAINDEX_T3
        - rad2mas

`drpangloss.legacy.savefits` re-exports `save` and the helpers above under
their old module name.
