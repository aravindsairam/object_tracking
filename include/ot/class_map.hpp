#pragma once

#include <string>
#include <unordered_set>
#include <vector>

namespace ot {

// Normalizes a model's native class indices into the abstract categories the
// rest of the system cares about (people + vehicles), and provides labels for
// the overlay. Built from a named preset matching the model that produced the
// detections (e.g. "coco", "waldo", "unidrone").
class ClassMap {
public:
    // Builds a preset map. Throws std::runtime_error on an unknown preset name.
    static ClassMap preset(const std::string& name);

    bool        keep(int class_id) const { return keep_.count(class_id) > 0; }
    std::string label(int class_id) const;
    const std::string& name() const { return name_; }

    // Kept native ids as a flat list (people/vehicles), for hot-loop argmax that
    // wants to iterate the ~6 kept classes directly instead of hashing all 80/91.
    const std::vector<int>& kept_ids() const { return kept_ids_; }

private:
    std::string                 name_;
    std::vector<std::string>    labels_;     // index = native class id
    std::unordered_set<int>     keep_;        // native ids that are people/vehicles
    std::vector<int>            kept_ids_;     // same ids as keep_, as a flat list
};

}  // namespace ot
