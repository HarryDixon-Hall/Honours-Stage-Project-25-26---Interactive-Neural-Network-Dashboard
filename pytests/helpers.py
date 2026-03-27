from collections.abc import Iterable


def walk_components(component):
    if component is None:
        return

    if isinstance(component, (str, bytes, int, float, bool)):
        yield component
        return

    yield component

    children = getattr(component, "children", None)
    if children is None:
        return

    if isinstance(children, (str, bytes, int, float, bool)):
        yield children
        return

    if isinstance(children, Iterable):
        for child in children:
            yield from walk_components(child)
        return

    yield from walk_components(children)


def collect_component_ids(component):
    component_ids = set()
    for node in walk_components(component):
        node_id = getattr(node, "id", None)
        if node_id:
            component_ids.add(node_id)
    return component_ids


def collect_text(component):
    text_chunks = []
    for node in walk_components(component):
        if isinstance(node, str):
            text_chunks.append(node)
    return " ".join(text_chunks)


def collect_prop_values(component, prop_name):
    values = []
    for node in walk_components(component):
        if hasattr(node, prop_name):
            value = getattr(node, prop_name)
            if value is not None:
                values.append(value)
    return values