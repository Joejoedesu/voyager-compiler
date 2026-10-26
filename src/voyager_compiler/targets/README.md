# Target-owned guides

Read only the guide for the target being changed. Each guide describes its bound
extension points, numerical/input-output contract, supported forms, padding and
storage decisions, and applicable validation. Registration and callable source
establish current behavior; dated evidence establishes only the exercised scope.
Shared contracts remain in [the compilation guide](../../../docs/compilation.md).

- [Default Voyager target](voyager/README.md)
- [Alpha](magnet_alpha/README.md)
- [Gemmini](gemmini/README.md)
- [Trainium](trainium/README.md)

A guide does not require a dedicated file for every hook. List actual bindings,
shared defaults and unbound optional hooks so readers can see where changes belong.
Do not repeat the shared adoption procedure or create a target completion tracker.
