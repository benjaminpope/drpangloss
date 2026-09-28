# `drpangloss.oifits_implaneia`

Legacy OIFITS helpers derived from [ImPlaneIA](https://github.com/anand0xff/ImPlaneIA),
working on its dictionary layout. Phases in these dictionaries are in
**degrees**. `save` queries SIMBAD for the target unless it is `"UNKNOWN"`.
New code should use [`drpangloss.oifits`](oifits.md), whose `write_oifits`
accepts the same dictionaries without network access, and read files with
[`OIData`](oidata.md). The module is not imported by `import drpangloss`.

## Functions

::: drpangloss.oifits_implaneia
    options:
      members:
        - rad2mas
        - GetWavelength
        - Format_STAINDEX_V2
        - Format_STAINDEX_T3
        - ApplyFlag
        - save
        - load
        - show
        - load_oifits
        - cp_indices
