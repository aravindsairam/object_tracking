#include "ot/class_map.hpp"

#include <stdexcept>

namespace ot {

std::string ClassMap::label(int class_id) const {
    if (class_id >= 0 && class_id < static_cast<int>(labels_.size())) {
        return labels_[class_id];
    }
    return "id" + std::to_string(class_id);
}

ClassMap ClassMap::preset(const std::string& name) {
    ClassMap m;
    m.name_ = name;

    if (name == "coco") {
        // COCO 80; people + vehicles only.
        m.labels_ = {
            "person","bicycle","car","motorcycle","airplane","bus","train","truck",
            "boat","traffic light","fire hydrant","stop sign","parking meter","bench"};
        // person, bicycle, car, motorcycle, bus, truck
        m.keep_ = {0, 1, 2, 3, 5, 7};
    } else if (name == "coco91") {
        // RF-DETR raw ONNX uses the 91-slot COCO category-id space (1-indexed;
        // id 0 is background). This differs from the contiguous 80-class "coco"
        // map above — verified against the exported model: labels=[1,300,91],
        // COCO_CLASSES{1:'person',2:'bicycle',...}.
        m.labels_ = {
            "background","person","bicycle","car","motorcycle","airplane","bus",
            "train","truck","boat","traffic light","fire hydrant"};
        // person, bicycle, car, motorcycle, bus, truck (1-indexed)
        m.keep_ = {1, 2, 3, 4, 6, 8};
    } else if (name == "waldo") {
        // WALDO30 (12 classes). people/vehicles: LightVehicle, Person, Bike, Truck, Bus.
        m.labels_ = {
            "LightVehicle","Person","Building","UPole","Boat","Bike",
            "Container","Truck","Gastank","Digger","SolarPanels","Bus"};
        m.keep_ = {0, 1, 5, 7, 11};
    } else if (name == "unidrone") {
        // UniDrone (7 classes). people/vehicles: person, bike, light_vehicle, truck, bus.
        m.labels_ = {"person","bike","light_vehicle","truck","bus","boat","digger"};
        m.keep_ = {0, 1, 2, 3, 4};
    } else {
        throw std::runtime_error("ClassMap: unknown preset '" + name +
                                 "' (expected coco | coco91 | waldo | unidrone)");
    }
    m.kept_ids_.assign(m.keep_.begin(), m.keep_.end());  // flat list for hot-loop argmax
    return m;
}

}  // namespace ot
